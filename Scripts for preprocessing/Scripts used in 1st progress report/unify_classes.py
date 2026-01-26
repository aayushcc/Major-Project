
from __future__ import annotations
import argparse
import os
import sys
import yaml
import shutil
import json
import re
from datetime import datetime
from collections import Counter, defaultdict
import difflib

# ---------------- DEFAULT CONFIG ----------------
# The canonical final class names and their order. (Change if you want a different target set.)
MASTER_NAMES = ["Bike", "Bus", "Car", "Cng", "Mini-Truck", "Rickshaw", "Truck"]

# Known small alias map (lowercase -> canonical lower)
# Add any aliases you know appear in your Roboflow exports here.
DEFAULT_MERGE_MAP = {
    # bikes / motorcycles
    "motorbike": "bike",
    "motorcycle": "bike",
    "motor": "bike",
    "moto": "bike",
    "scooter": "bike",
    "bikes": "bike",
    # auto / rickshaw
    "auto": "rickshaw",
    "autorickshaw": "rickshaw",
    "auto-rickshaw": "rickshaw",
    "tuk-tuk": "rickshaw",
    # trucks
    "lorry": "truck",
    "pickup": "truck",
    "mini truck": "mini-truck",
    "mini-trucks": "mini-truck",
    # cars
    "car": "car",
    "cars": "car",
    # buses
    "bus": "bus",
    "buses": "bus",
    # CNG (compressed natural gas three-wheeler common in some regions)
    "cng": "cng",
}

# classes we want to drop entirely if present
DEFAULT_REMOVE_SET = {"red-light", "green-light", "yellow-light", "cycle", "people", "person"}

# Label file extensions and splits we'll search for by default
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ["train", "valid", "val", "test"]

# -------------------------------------------------


def norm_name(s: str) -> str:
    """Lower-case, strip and remove common punctuation. Useful for matching.
    (Removes characters other than letters, numbers, space and hyphen.)"""
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9\- ]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def load_data_yaml(dataset_path: str):
    ypath = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(ypath):
        return None, None
    with open(ypath, "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names", []) or []
    return data, list(names)


def find_label_files(dataset_path: str):
    """Discover label files under dataset path. Returns list of (split, label_path)."""
    found = []
    # first look for conventional splits
    for split in SPLITS:
        labels_dir = os.path.join(dataset_path, split, "labels")
        if os.path.isdir(labels_dir):
            for fn in os.listdir(labels_dir):
                if not fn.lower().endswith(".txt"):
                    continue
                found.append((split, os.path.join(labels_dir, fn)))
    # fallback: scan subfolders for any labels directory
    if not found:
        for root, dirs, files in os.walk(dataset_path):
            if os.path.basename(root).lower() == "labels":
                for fn in files:
                    if fn.lower().endswith(".txt"):
                        # guess split name as parent folder of 'labels'
                        split_guess = os.path.basename(os.path.dirname(root))
                        found.append((split_guess, os.path.join(root, fn)))
    return found


def gather_used_indices(dataset_path: str):
    """Return a Counter of indices used across all label files."""
    files = find_label_files(dataset_path)
    counts = Counter()
    for _split, fp in files:
        try:
            with open(fp, "r") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    parts = s.split()
                    try:
                        idx = int(float(parts[0]))
                    except Exception:
                        continue
                    counts[idx] += 1
        except Exception:
            continue
    return counts


def build_proposed_mapping(original_names: list[str], counts: Counter, master_names: list[str],
                           merge_map: dict[str, str], remove_set: set[str], fuzzy_cutoff: float = 0.75):
    """
    Return mapping old_idx -> dict with keys: new_idx (int|None), reason (str), score (float|None)
    - new_idx is index into master_names or None (means drop)
    """
    master_lower = [norm_name(n) for n in master_names]
    master_map = {ml: i for i, ml in enumerate(master_lower)}
    # normalize provided merge_map and remove_set
    merge_map_norm = {norm_name(k): norm_name(v) for k, v in merge_map.items()}
    remove_norm = {norm_name(x) for x in remove_set}

    mapping = {}
    max_idx = max(max(counts.keys()) if counts else -1, len(original_names) - 1)
    for old_idx in range(max_idx + 1):
        # original name may not exist in YAML if labels reference larger index
        orig_name = original_names[old_idx] if old_idx < len(original_names) else f"<index_{old_idx}_no_name>"
        low = norm_name(orig_name)
        reason = ""
        score = None
        new_idx = None

        if low in remove_norm:
            reason = "explicit-remove"
            new_idx = None
        elif low in merge_map_norm:
            # merge_map tells canonical lower name -> e.g. 'bike'
            can = merge_map_norm[low]
            if can in master_map:
                new_idx = master_map[can]
                reason = f"merge_map->{master_names[new_idx]}"
            else:
                reason = f"merge_map->{can} (not in master)"
                new_idx = None
        elif low in master_map:
            new_idx = master_map[low]
            reason = f"direct-match->{master_names[new_idx]}"
        else:
            # check fuzzy match to master names
            # use difflib to get a decent candidate
            cand = difflib.get_close_matches(low, master_lower, n=1, cutoff=fuzzy_cutoff)
            if cand:
                can = cand[0]
                new_idx = master_map[can]
                # compute score using SequenceMatcher real ratio
                score = difflib.SequenceMatcher(None, low, can).ratio()
                reason = f"fuzzy->{master_names[new_idx]}"
            else:
                # If name is gibberish (pure digits or <index_> placeholder) treat as "unknown"
                if re.fullmatch(r"\d+", low) or low.startswith("index_") or low.startswith("link"):
                    reason = "unknown-garbage"
                    new_idx = None
                else:
                    reason = "unknown"
                    new_idx = None

        mapping[old_idx] = {"new_idx": new_idx, "reason": reason, "score": score, "orig_name": orig_name,
                             "count": counts.get(old_idx, 0)}

    return mapping


def backup_dataset(dataset_path: str, timestamp: str):
    # backup data.yaml
    y = os.path.join(dataset_path, "data.yaml")
    if os.path.exists(y):
        bak = os.path.join(dataset_path, f"data.yaml.bak.{timestamp}")
        if not os.path.exists(bak):
            shutil.copy2(y, bak)
    # backup label folders (copy only once)
    all_labels = find_label_files(dataset_path)
    if not all_labels:
        return
    backup_root = os.path.join(dataset_path, f"labels_backup_{timestamp}")
    if os.path.exists(backup_root):
        return
    os.makedirs(backup_root, exist_ok=True)
    # recreate split/labels structure
    for split, fp in all_labels:
        rel_dir = os.path.join(split, "labels")
        target_dir = os.path.join(backup_root, rel_dir)
        os.makedirs(target_dir, exist_ok=True)
    # copy files
    for split, fp in all_labels:
        fname = os.path.basename(fp)
        rel_dir = os.path.join(split, "labels")
        target_dir = os.path.join(backup_root, rel_dir)
        try:
            shutil.copy2(fp, os.path.join(target_dir, fname))
        except Exception:
            pass


def apply_mapping_to_labels(dataset_path: str, mapping: dict, apply_changes: bool = False):
    files = find_label_files(dataset_path)
    total_before = 0
    total_after = 0
    removed = 0
    changed = 0
    per_old_removed = defaultdict(int)

    for split, fp in files:
        try:
            with open(fp, "r") as f:
                lines = [L.strip() for L in f if L.strip()]
        except Exception:
            continue
        out_lines = []
        for L in lines:
            parts = L.split()
            try:
                old_idx = int(float(parts[0]))
            except Exception:
                # corrupt line -> skip
                continue
            total_before += 1
            info = mapping.get(old_idx, {"new_idx": None})
            new_idx = info["new_idx"]
            if new_idx is None:
                removed += 1
                per_old_removed[old_idx] += 1
                continue
            if new_idx != old_idx:
                changed += 1
            out_lines.append(" ".join([str(new_idx)] + parts[1:]))
        total_after += len(out_lines)
        if apply_changes:
            try:
                with open(fp, "w") as f:
                    f.write("\n".join(out_lines) + ("\n" if out_lines else ""))
            except Exception as e:
                print(f"Failed to write {fp}: {e}")

    return {"total_before": total_before, "total_after": total_after,
            "removed": removed, "changed": changed, "per_old_removed": per_old_removed}


def write_unified_data_yaml(dataset_path: str, data: dict, master_names: list[str]):
    ypath = os.path.join(dataset_path, "data.yaml")
    # preserve other fields but force names & nc
    data_out = dict(data) if data else {}
    data_out["names"] = master_names
    data_out["nc"] = len(master_names)
    try:
        with open(ypath, "w") as f:
            yaml.safe_dump(data_out, f, sort_keys=False)
        return True
    except Exception as e:
        print(f"Failed to write data.yaml: {e}")
        return False


def process_dataset(dataset_path: str, master_names: list[str], merge_map: dict, remove_set: set,
                    apply_changes: bool = False, interactive: bool = False, yes: bool = False):
    print(f"\n--- Processing: {dataset_path} ---")
    data, orig_names = load_data_yaml(dataset_path)
    if data is None:
        print("  ERROR: data.yaml not found. Skipping.")
        return

    counts = gather_used_indices(dataset_path)
    used_indices_sorted = sorted(counts.keys())
    print(f"  Found used label indices: {used_indices_sorted[:30]}{'...' if len(used_indices_sorted)>30 else ''}")

    mapping = build_proposed_mapping(orig_names, counts, master_names, merge_map, remove_set)

    # Print proposed mapping summary
    print("  Proposed mapping (old_idx : orig_name [count] -> new_idx / reason):")
    for i in sorted(mapping.keys()):
        m = mapping[i]
        nj = m["new_idx"]
        nm = f"{nj} ({master_names[nj]})" if nj is not None else "None"
        score = f", score={m['score']:.2f}" if m.get('score') else ""
        print(f"    {i}: '{m['orig_name']}' [{m['count']}] -> {nm}  [{m['reason']}{score}]")

    # interactive resolution of unknowns
    if interactive:
        changed_any = False
        for i, m in mapping.items():
            if m['new_idx'] is None:
                # ask user what to do
                print(f"\nIndex {i}: '{m['orig_name']}' occurs {m['count']} times. Proposed: DROP (reason={m['reason']}).\n")
                print("Choose action:")
                print("  0) Drop this class (remove its label lines)")
                for j, cname in enumerate(master_names):
                    print(f"  {j+1}) Map to {j} ({cname})")
                ans = input("Enter number (or Enter to keep DROP): ").strip()
                if ans == "":
                    continue
                try:
                    ai = int(ans)
                    if ai == 0:
                        mapping[i]['new_idx'] = None
                        mapping[i]['reason'] = 'user-drop'
                    elif 1 <= ai <= len(master_names):
                        mapping[i]['new_idx'] = ai - 1
                        mapping[i]['reason'] = 'user-map'
                    changed_any = True
                except Exception:
                    print("Invalid input, skipping.")
        if changed_any:
            print("Interactive changes recorded.")

    # If apply_changes then create backups and write
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if apply_changes:
        if not yes:
            resp = input("You passed --apply but not --yes. Confirm write changes and create backups? (y/N): ").strip().lower()
            if resp != 'y':
                print("Aborting apply.")
                return
        print("  Making backups...")
        backup_dataset(dataset_path, timestamp)

    stats = apply_mapping_to_labels(dataset_path, mapping, apply_changes=apply_changes)

    if apply_changes:
        # write unified data.yaml with master names
        write_unified_data_yaml(dataset_path, data, master_names)
        # save mapping file for audit
        map_out = {str(k): mapping[k] for k in mapping}
        with open(os.path.join(dataset_path, f"mapping_unify_{timestamp}.json"), "w") as jf:
            json.dump(map_out, jf, indent=2)

    # summary
    print("  Summary:")
    print(f"    label_lines_before = {stats['total_before']}")
    print(f"    label_lines_after  = {stats['total_after']}")
    print(f"    removed_labels     = {stats['removed']}")
    print(f"    reindexed_labels   = {stats['changed']}")
    if stats['per_old_removed']:
        print("    Removed per original index (top 10):")
        for idx, cnt in sorted(stats['per_old_removed'].items(), key=lambda x: -x[1])[:10]:
            name = mapping.get(idx, {}).get('orig_name', f'idx{idx}')
            print(f"      {idx} '{name}': {cnt}")

    # warn about unknowns that will be dropped
    unknowns = [(i, mapping[i]['orig_name']) for i in mapping if mapping[i]['new_idx'] is None and mapping[i]['count'] > 0]
    if unknowns:
        print("  WARNING: these original classes map to None (will be dropped):")
        for i, name in unknowns:
            print(f"    {i}: '{name}'")

    if not apply_changes:
        print("  (dry-run) No files changed. Re-run with --apply --yes to write files and backups.")
    else:
        print("  Changes applied. Backups saved (data.yaml.bak.* and labels_backup_*). Mapping written to mapping_unify_*.json")


def parse_args():
    p = argparse.ArgumentParser(description="Safer unify classes across Roboflow/YOLO datasets.")
    p.add_argument("datasets", nargs="*", help="Paths to dataset folders (if empty, script will try '1','2','3','4' relative to script parent)")
    p.add_argument("--apply", action="store_true", help="Actually write changes and make backups. Otherwise dry-run only.")
    p.add_argument("--yes", action="store_true", help="When used with --apply, skip interactive confirmation.")
    p.add_argument("--interactive", action="store_true", help="Ask interactively about ambiguous/unknown classes.")
    p.add_argument("--master", help="Path to a JSON/YAML file that defines MASTER_NAMES (overrides built-in master)")
    p.add_argument("--list-mapping", action="store_true", help="Print default merge/remove maps and exit.")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # load alternate master if requested
    master = MASTER_NAMES
    if args.master:
        if os.path.exists(args.master):
            with open(args.master, 'r') as f:
                try:
                    parsed = yaml.safe_load(f)
                    if isinstance(parsed, list):
                        master = parsed
                    elif isinstance(parsed, dict) and 'names' in parsed:
                        master = parsed['names']
                except Exception as e:
                    print(f"Failed to load master names from {args.master}: {e}")
                    sys.exit(1)
        else:
            print(f"Master file {args.master} not found.")
            sys.exit(1)

    if args.list_mapping:
        print("Default MERGE_MAP (aliases -> canonical):")
        for k, v in DEFAULT_MERGE_MAP.items():
            print(f"  {k} -> {v}")
        print("Default REMOVE_SET:")
        for x in DEFAULT_REMOVE_SET:
            print(f"  {x}")
        sys.exit(0)

    if args.datasets:
        targets = args.datasets
    else:
        base = os.path.dirname(os.path.abspath(__file__))
        targets = [os.path.join(base, d) for d in ["1", "2", "3", "4"]]

    for t in targets:
        if not os.path.exists(t):
            print(f"Skipping not-found: {t}")
            continue
        process_dataset(t, master, DEFAULT_MERGE_MAP, DEFAULT_REMOVE_SET,
                        apply_changes=args.apply, interactive=args.interactive, yes=args.yes)

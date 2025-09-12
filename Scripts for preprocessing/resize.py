import os
from PIL import Image
import shutil

# Target size
TARGET_SIZE = 640

# Paths
DATASET_DIR = "haha"
SETS = ["train", "test", "valid"]

def pad_and_adjust(image_path, label_path, save_image_path, save_label_path):
    # Open image
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    # Scale to fit within 640x640, keep aspect ratio
    scale = min(TARGET_SIZE / orig_w, TARGET_SIZE / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized_img = img.resize((new_w, new_h), Image.LANCZOS)

    # Create new black image
    new_img = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))

    # Calculate padding offsets
    pad_x = (TARGET_SIZE - new_w) // 2
    pad_y = (TARGET_SIZE - new_h) // 2

    # Paste resized image into black canvas
    new_img.paste(resized_img, (pad_x, pad_y))
    new_img.save(save_image_path)

    # Adjust labels
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x_center, y_center, w, h = parts
            x_center = float(x_center) * orig_w * scale + pad_x
            y_center = float(y_center) * orig_h * scale + pad_y
            w = float(w) * orig_w * scale
            h = float(h) * orig_h * scale

            # Normalize again to 0–1
            x_center /= TARGET_SIZE
            y_center /= TARGET_SIZE
            w /= TARGET_SIZE
            h /= TARGET_SIZE

            new_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        with open(save_label_path, "w") as f:
            f.writelines(new_lines)

def process_dataset():
    for set_name in SETS:
        img_dir = os.path.join(DATASET_DIR, set_name, "images")
        lbl_dir = os.path.join(DATASET_DIR, set_name, "labels")

        for img_file in os.listdir(img_dir):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(img_dir, img_file)
            label_path = os.path.join(lbl_dir, os.path.splitext(img_file)[0] + ".txt")

            save_img_path = img_path  # overwrite original
            save_label_path = label_path

            pad_and_adjust(img_path, label_path, save_img_path, save_label_path)

if __name__ == "__main__":
    process_dataset()

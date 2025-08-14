import os
import shutil
from PIL import Image
import imagehash

# Path to your train/images folder
image_dir = "/home/aayush/Documents/Untitled Folder/train/images"
output_dir = "near_duplicates"
os.makedirs(output_dir, exist_ok=True)

hash_dict = {}
near_dup_count = 0
threshold = 5  # Lower = stricter, 0 = exact match, 5 = fairly similar

for filename in os.listdir(image_dir):
    filepath = os.path.join(image_dir, filename)

    if not os.path.isfile(filepath):
        continue

    try:
        img = Image.open(filepath).convert("RGB")
        phash = imagehash.phash(img)
    except Exception as e:
        print(f"⚠️ Error reading {filename}: {e}")
        continue

    found_similar = False
    for existing_hash in hash_dict:
        if phash - existing_hash <= threshold:
            shutil.move(filepath, os.path.join(output_dir, filename))
            near_dup_count += 1
            found_similar = True
            break

    if not found_similar:
        hash_dict[phash] = filename

print(f"✅ Found and moved {near_dup_count} near-duplicates to '{output_dir}'")

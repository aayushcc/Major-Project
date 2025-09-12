import os
import shutil

# Dataset folder names
dataset_folders = ["1", "2", "3", "4"]

# Target unified structure
target_structure = [
    ("train/images", "train/images"),
    ("train/labels", "train/labels"),
    ("test/images", "test/images"),
    ("test/labels", "test/labels"),
    ("valid/images", "valid/images"),
    ("valid/labels", "valid/labels")
]

# Create the target folders if they don't exist
for _, target in target_structure:
    os.makedirs(target, exist_ok=True)

# Move files from each dataset into the unified folder
for dataset in dataset_folders:
    for src_subfolder, target_subfolder in target_structure:
        src_path = os.path.join(dataset, src_subfolder)
        if os.path.exists(src_path):
            for file_name in os.listdir(src_path):
                src_file = os.path.join(src_path, file_name)
                dst_file = os.path.join(target_subfolder, file_name)
                shutil.move(src_file, dst_file)

print("✅ All datasets merged successfully!")

import cv2
import os
import yaml

# Load class names from data.yaml
with open("/home/aayush/Documents/Untitled Folder/1/data.yaml", 'r') as f:
    data = yaml.safe_load(f)
CLASS_NAMES = data['names']  # This should be a list of names like ['cat', 'dog', 'car']


# ==== CONFIGURATION ====
images_dir = "/home/aayush/Documents/Untitled Folder/1/test/images"
labels_dir = "/home/aayush/Documents/Untitled Folder/1/test/labels"
output_dir = "output_visualized"
os.makedirs(output_dir, exist_ok=True)

# Give different colors for different classes
COLORS = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]

# ==== MAIN ====
for img_file in os.listdir(images_dir):
    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(images_dir, img_file)
    label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")

    # Read image
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]

    # Skip if no label
    if not os.path.exists(label_path):
        continue

    # Read YOLO label file
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id, x_center, y_center, width, height = map(float, parts)

            # Convert from normalized to pixel coordinates
            x_center *= w
            y_center *= h
            width *= w
            height *= h

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            color = COLORS[int(cls_id) % len(COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label_text = CLASS_NAMES[int(cls_id)] if int(cls_id) < len(CLASS_NAMES) else str(int(cls_id))
            cv2.putText(img, label_text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


    # Save visualized image
    out_path = os.path.join(output_dir, img_file)
    cv2.imwrite(out_path, img)

print(f"Visualization saved in '{output_dir}'")

import cv2


image_path = "/home/aayush/Documents/Dataset Manipulation/Edited/E4 Post Merge(Fixed + Removed Classes)/test/images/frame74_jpg.rf.d39b76ce4436189051136da1ca5d8f37.jpg"
label_path = "/home/aayush/Documents/Dataset Manipulation/Edited/E4 Post Merge(Fixed + Removed Classes)/test/labels/frame74_jpg.rf.d39b76ce4436189051136da1ca5d8f37.txt"

class_names = {
    "Cng": 0,
    "Rickshaw": 1,
    "Car": 2,
    "Bus": 3,
    "Bike": 4,
    "Mini-Truck": 5,
    "Truck": 6
}

# build an index->name mapping for lookup by numeric class id
idx_to_name = {v: k for k, v in class_names.items()}

img = cv2.imread(image_path)
h, w = img.shape[:2]

with open(label_path, "r") as f:
    lines = f.readlines()

    for line in lines:
        cls, xc, yc, bw, bh = map(float, line.strip().split())

        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # safe label lookup: map numeric class id to name, fallback if missing
        label = idx_to_name.get(int(cls), f"Class {int(cls)}")
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow("YOLO Label Check", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

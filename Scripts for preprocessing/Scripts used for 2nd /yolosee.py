import cv2
import os


image_path = "/home/aayush/Documents/Dataset Manipulation/Edited/E2 copy (Removed classes)/1-LOC2-0700-0800-DL.v1i.voc/frame0_jpg.rf.263ac5c11ff962697a10265a32a937f6.jpg"
label_path = "/home/aayush/Documents/Dataset Manipulation/Edited/E2 copy (Removed classes)/1-LOC2-0700-0800-DL.v1i.voc/frame0_jpg.rf.263ac5c11ff962697a10265a32a937f6.txt"
class_names = [
    "Cng": 0,
    "Rickshaw": 1,
    "Car": 2,
    "Bus": 3,
    "Bike": 4,
    "People": 5,
    "Mini-Truck": 6,
    "Cycle": 7,
    "Truck": 8
]

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
    label = class_names[int(cls)]
    cv2.putText(img, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow("YOLO Label Check", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

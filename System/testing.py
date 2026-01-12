import cv2
import numpy as np
from ultralytics import YOLO
import time

MODEL_PATH = "yolo11s.pt"
model = YOLO(MODEL_PATH)

#DEFINE SINGLE LANE ROI
ROI = np.array([(125, 492), (891, 490), (890, 326), (596, 96), (292, 99), (122, 490)])

# Vehicle classes to track
VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

# Green light limits
G_MIN = 5
G_MAX = 45
CYCLE_BUDGET = 60     # Maximum adjustable time per cycle
CYCLE_INTERVAL = 10   # Recalculate green light every 10 seconds

# Helper: check point inside polygon
def point_in_poly(pt, poly):
    return cv2.pointPolygonTest(poly, pt, False) >= 0

# Helper: get bounding box center
def bbox_centroid(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

# Video input
video_path = "/home/sumankhatri/Videos/lane1.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps, "Width:", w, "Height:", h)


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter("output.mp4", fourcc, fps, (w, h))

if not writer.isOpened():
    print("❌ ERROR: VideoWriter failed to open!")
    print("Check codec, file path, or permissions.")
    exit()
else:
    print("✅ VideoWriter opened successfully!")

last_cycle_time = time.time()
green_time = G_MIN  # initial green time

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model
    results = model(frame, conf=0.4)
    r = results[0]

    vehicles = []
    for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
        name = model.names[int(cls)]
        if name in VEHICLE_CLASSES:
            vehicles.append(box.cpu().numpy().astype(int))

    # Count vehicles inside ROI
    lane_count = 0
    for box in vehicles:
        cx, cy = bbox_centroid(box)
        if point_in_poly((cx, cy), ROI):
            lane_count += 1

    # Recalculate green time every CYCLE_INTERVAL seconds
    now = time.time()
    if now - last_cycle_time >= CYCLE_INTERVAL:
        last_cycle_time = now

        # Traffic logic (simple adaptive logic)
        if lane_count == 0:
            green_time = G_MIN
        else:
            proportion = min(lane_count / 10, 1)  # normalize
            green_time = int(G_MIN + proportion * CYCLE_BUDGET)

        green_time = max(G_MIN, min(G_MAX, green_time))

    # ---------------- DRAW EVERYTHING ----------------

    # ROI
    cv2.polylines(frame, [ROI], True, (0, 255, 255), 2)

    # Display ROI info
    cv2.putText(frame,
                f"Lane Vehicles: {lane_count}",
                (50, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    cv2.putText(frame,
                f"Green Time: {green_time}s",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # Draw boxes
    for box in vehicles:
        x1, y1, x2, y2 = box
        cx, cy = bbox_centroid(box)
        color = (0, 255, 0) if point_in_poly((cx, cy), ROI) else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    writer.write(frame)
    cv2.imshow("Single Lane Traffic Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

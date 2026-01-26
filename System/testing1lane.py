import cv2
import numpy as np
import time

from t_detector import VehicleDetector
from t_utils import point_in_poly, bbox_centroid

# ---------------- CONFIG ----------------
MODEL_PATH = "yolo11s.pt"
VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]
# VEHICLE_CLASSES = ["Cng","Rickshaw","Car","Bus","Bike","Mini-Truck","Truck"]


G_MIN, G_MAX = 5, 45
CYCLE_BUDGET = 60
CYCLE_INTERVAL = 5

LANE = {
    "video": "System\B_lane1.mp4",
    "ROI": np.array([
        (3, 419), (1, 712), (1275, 695),
        (1273, 489), (1103, 174), (441, 144), (4, 419)
    ])
}

# ---------------- INITIALIZATION ----------------
detector = VehicleDetector(MODEL_PATH, VEHICLE_CLASSES)

cap = cv2.VideoCapture(LANE["video"])
if not cap.isOpened():
    raise RuntimeError("Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS)
fps = fps if fps > 0 else 30.0
frame_duration = 1.0 / fps

green_time = G_MIN
last_cycle_time = time.time()

print("🚦 TRUE VIDEO-TIME SYNCHRONIZED MODE")
print(f"🎞 FPS: {fps}")

start_wall = time.time()
start_video = start_wall

frame_index = 0
processed_frames = 0

# --- smoothing state (important for stability)
smoothed_count = 0
alpha = 0.3

# ---------------- MAIN LOOP ----------------
while True:
    target_time = start_video + frame_index * frame_duration
    now = time.time()

    # If we're too far behind → skip frame
    if now > target_time + frame_duration:
        cap.grab()
        frame_index += 1
        continue

    ret, frame = cap.read()
    if not ret:
        break

    processed_frames += 1
    frame_index += 1

    # -------- Detection (YOLO handles resizing internally) --------
    vehicles = detector.detect(frame)

    # -------- Vehicle counting --------
    raw_count = sum(
        1 for box in vehicles
        if point_in_poly(bbox_centroid(box), LANE["ROI"])
    )

    # Smooth the count (prevents flickering green time)
    smoothed_count = int(alpha * raw_count + (1 - alpha) * smoothed_count)
    lane_count = smoothed_count

    # -------- Green time logic --------
    now = time.time()
    if now - last_cycle_time >= CYCLE_INTERVAL:
        last_cycle_time = now
        proportion = min(lane_count / 10, 1)
        green_time = int(G_MIN + proportion * CYCLE_BUDGET)
        green_time = max(G_MIN, min(G_MAX, green_time))

    # -------- Drawing --------
    cv2.polylines(frame, [LANE["ROI"]], True, (0, 255, 255), 2)

    cv2.putText(frame, f"Vehicles: {lane_count}", (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(frame, f"Green Time: {green_time}s", (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    for box in vehicles:
        x1, y1, x2, y2 = box
        color = (0, 255, 0) if point_in_poly(
            bbox_centroid(box), LANE["ROI"]
        ) else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.imshow("Lane 1 (Video-Time Sync)", frame)

    # Sleep until exact display time
    sleep = target_time - time.time()
    if sleep > 0:
        time.sleep(sleep)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
end_wall = time.time()

cap.release()
cv2.destroyAllWindows()

print("\n✅ Done")
print(f"🧮 Frames processed: {processed_frames}")
print(f"⏱ Wall time: {end_wall - start_wall:.2f}s")
print(f"🎥 Video time: {frame_index / fps:.2f}s")

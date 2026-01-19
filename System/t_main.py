import cv2
import numpy as np
import time

from t_detector import VehicleDetector
from t_utils import point_in_poly, bbox_centroid, read_latest_frame

MODEL_PATH = "yolo11s.pt"
VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

G_MIN, G_MAX = 5, 45
CYCLE_BUDGET = 60
CYCLE_INTERVAL = 5
TARGET_FPS = 4   # YOLO runs 4 times/sec

LANES = [
    {"video": "/home/sumankhatri/Videos/lane1.mp4",
     "ROI": np.array([(125, 492), (891, 490), (890, 326), (596, 96), (292, 99), (122, 490)])},
    {"video": "/home/sumankhatri/Videos/lane2.mp4",
     "ROI": np.array([(100, 500), (900, 500), (900, 350), (600, 100), (250, 100), (100, 500)])},
    {"video": "/home/sumankhatri/Videos/lane3.mp4",
     "ROI": np.array([(150, 480), (880, 480), (880, 320), (580, 90), (300, 90), (150, 480)])},
    {"video": "/home/sumankhatri/Videos/lane4.mp4",
     "ROI": np.array([(130, 490), (870, 490), (870, 330), (590, 95), (280, 95), (130, 490)])},
]

detector = VehicleDetector(MODEL_PATH, VEHICLE_CLASSES)

for lane in LANES:
    lane["cap"] = cv2.VideoCapture(lane["video"])
    lane["fps"] = lane["cap"].get(cv2.CAP_PROP_FPS) or 20.0
    lane["w"] = int(lane["cap"].get(cv2.CAP_PROP_FRAME_WIDTH))
    lane["h"] = int(lane["cap"].get(cv2.CAP_PROP_FRAME_HEIGHT))

    lane["resize_w"] = lane["w"] // 2
    lane["resize_h"] = lane["h"] // 2

    lane["writer"] = cv2.VideoWriter(
        lane["video"].replace(".mp4", "_out.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        lane["fps"],
        (lane["w"], lane["h"])
    )

    lane["frame_counter"] = 0
    lane["skip_frames"] = max(int(lane["fps"] / TARGET_FPS), 1)
    lane["last_detection"] = []

    lane["green_time"] = G_MIN
    lane["last_cycle_time"] = time.time()

print("✅ All lanes initialized. Real-time processing active.")

# MAIN LOOP
while True:
    all_done = True

    for idx, lane in enumerate(LANES):
        frame = read_latest_frame(lane["cap"])
        if frame is None:
            continue

        all_done = False
        lane["frame_counter"] += 1

        # -------- Detection (only on selected frames) --------
        if lane["frame_counter"] % lane["skip_frames"] == 0:
            vehicles = detector.detect(
                frame,
                (lane["resize_w"], lane["resize_h"]),
                (lane["w"], lane["h"])
            )
            lane["last_detection"] = vehicles
        else:
            vehicles = lane["last_detection"]

        # -------- Vehicle counting --------
        lane_count = sum(
            1 for box in vehicles
            if point_in_poly(bbox_centroid(box), lane["ROI"])
        )

        # -------- Green time logic --------
        now = time.time()
        if now - lane["last_cycle_time"] >= CYCLE_INTERVAL:
            lane["last_cycle_time"] = now
            proportion = min(lane_count / 10, 1)
            lane["green_time"] = int(G_MIN + proportion * CYCLE_BUDGET)
            lane["green_time"] = max(G_MIN, min(G_MAX, lane["green_time"]))

        # -------- Draw --------
        cv2.polylines(frame, [lane["ROI"]], True, (0, 255, 255), 2)
        cv2.putText(frame, f"Lane Vehicles: {lane_count}", (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Green Time: {lane['green_time']}s", (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for box in vehicles:
            x1, y1, x2, y2 = box
            color = (0, 255, 0) if point_in_poly(bbox_centroid(box), lane["ROI"]) else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        lane["writer"].write(frame)
        cv2.imshow(f"Lane {idx + 1}", frame)

    if all_done:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
for lane in LANES:
    lane["cap"].release()
    lane["writer"].release()

cv2.destroyAllWindows()


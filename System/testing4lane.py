import cv2
import numpy as np
import time

from t_detector import VehicleDetector
from t_utils import point_in_poly, bbox_centroid

# ---------------- CONFIG ----------------
MODEL_PATH = "yolo11s.pt"
VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

G_MIN = 5
G_MAX = 45

LANES = [
    {
        "name": "EAST",
        "video": "/home/sumankhatri/Videos/B_lane1.mp4",
        "ROI": np.array([(3, 419), (1, 712), (1275, 695),
        (1273, 489), (1103, 174), (441, 144), (4, 419)])
    },
    {
        "name": "SOUTH",
        "video": "/home/sumankhatri/Videos/B_lane2.mp4",
        "ROI": np.array([(3, 419), (1, 712), (1275, 695),
        (1273, 489), (1103, 174), (441, 144), (4, 419)])
    },
    {
        "name": "WEST",
        "video": "/home/sumankhatri/Videos/B_lane3.mp4",
        "ROI": np.array([(3, 419), (1, 712), (1275, 695),
        (1273, 489), (1103, 174), (441, 144), (4, 419)])
    },
    {
        "name": "NORTH",
        "video": "/home/sumankhatri/Videos/B_lane4.mp4",
        "ROI": np.array([(3, 419), (1, 712), (1275, 695),
        (1273, 489), (1103, 174), (441, 144), (4, 419)])
    }
]

# ---------------- INIT ----------------
detector = VehicleDetector(MODEL_PATH, VEHICLE_CLASSES)

for lane in LANES:
    lane["cap"] = cv2.VideoCapture(lane["video"])
    lane["green_time"] = G_MIN
    lane["last_green_start"] = time.time()
    lane["vehicle_count"] = 0

current_lane_idx = 0  # EAST starts green
LANES[current_lane_idx]["last_green_start"] = time.time()

print("🚦 Traffic system started")

# ---------------- FUNCTIONS ----------------
def compute_green_time(count):
    ratio = min(count / 10.0, 1.0)
    return int(G_MIN + ratio * (G_MAX - G_MIN))

def detect_lane(lane):
    ret, frame = lane["cap"].read()
    if not ret:
        return 0

    boxes = detector.detect(frame)
    count = 0

    for box in boxes:
        if point_in_poly(bbox_centroid(box), lane["ROI"]):
            count += 1

    return count

# ---------------- MAIN LOOP ----------------
while True:
    now = time.time()
    current_lane = LANES[current_lane_idx]

    elapsed = now - current_lane["last_green_start"]

    # ---- switch when green ends ----
    if elapsed >= current_lane["green_time"]:
        # move to next direction
        current_lane_idx = (current_lane_idx + 1) % len(LANES)
        next_lane = LANES[current_lane_idx]

        # detect vehicles for NEW lane
        count = detect_lane(next_lane)
        next_lane["vehicle_count"] = count
        next_lane["green_time"] = compute_green_time(count)
        next_lane["last_green_start"] = now

        print(
            f"➡️ {next_lane['name']} GREEN | "
            f"Vehicles: {count} | "
            f"Time: {next_lane['green_time']}s"
        )

    # ---- DISPLAY ----
    for idx, lane in enumerate(LANES):
        ret, frame = lane["cap"].read()
        if not ret:
            continue

        cv2.polylines(frame, [lane["ROI"]], True, (255, 255, 0), 2)

        if idx == current_lane_idx:
            remaining = max(
                0,
                int(lane["green_time"] - (time.time() - lane["last_green_start"]))
            )
            cv2.putText(frame, f"{lane['name']} GREEN {remaining}s",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"{lane['name']} RED",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

        cv2.imshow(lane["name"], frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    time.sleep(0.05)

# ---------------- CLEANUP ----------------
for lane in LANES:
    lane["cap"].release()

cv2.destroyAllWindows()

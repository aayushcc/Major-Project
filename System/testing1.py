import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch

# ---------------- CUDA CHECK ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {DEVICE}")

# ---------------- MODEL ----------------
MODEL_PATH = "yolo11s.pt"
model = YOLO(MODEL_PATH)
model.to(DEVICE)

# ---------------- CONFIG ----------------
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

VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

G_MIN, G_MAX = 5, 45
CYCLE_BUDGET = 60
CYCLE_INTERVAL = 10
TARGET_FPS = 5

# ---------------- HELPERS ----------------
def point_in_poly(pt, poly):
    return cv2.pointPolygonTest(poly, pt, False) >= 0

def bbox_centroid(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def read_latest_frame(cap, max_drop=8):
    """
    Always returns the most recent frame.
    Older frames are dropped to prevent lag.
    """
    frame = None
    for _ in range(max_drop):
        ret, f = cap.read()
        if not ret:
            break
        frame = f
    return frame

# ---------------- INIT LANES ----------------
for lane in LANES:
    lane["cap"] = cv2.VideoCapture(lane["video"])
    lane["fps"] = lane["cap"].get(cv2.CAP_PROP_FPS) or 20.0
    lane["w"] = int(lane["cap"].get(cv2.CAP_PROP_FRAME_WIDTH))
    lane["h"] = int(lane["cap"].get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Downscale for faster inference
    lane["resize_w"] = lane["w"] // 2
    lane["resize_h"] = lane["h"] // 2

    lane["writer"] = cv2.VideoWriter(
        lane["video"].replace(".mp4", "_out.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        lane["fps"],
        (lane["w"], lane["h"])
    )

    lane["green_time"] = G_MIN
    lane["last_cycle_time"] = time.time()
    lane["last_detection"] = []
    lane["frame_counter"] = 0
    lane["skip_frames"] = max(int(lane["fps"] / TARGET_FPS), 1)

print("✅ All lanes initialized. Real-time CUDA inference active.")

# ---------------- MAIN LOOP ----------------
while True:
    all_done = True

    for idx, lane in enumerate(LANES):
        frame = read_latest_frame(lane["cap"])
        if frame is None:
            continue

        all_done = False
        lane["frame_counter"] += 1

        small_frame = cv2.resize(frame, (lane["resize_w"], lane["resize_h"]))

        # ---------- YOLO INFERENCE ----------
        if lane["frame_counter"] % lane["skip_frames"] == 0:
            results = model(
                small_frame,
                conf=0.4,
                device=DEVICE,
                half=True,     # FP16 on GPU
                verbose=False
            )

            vehicles = []
            r = results[0]

            if r.boxes is not None:
                for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                    if model.names[int(cls)] in VEHICLE_CLASSES:
                        x1, y1, x2, y2 = box.cpu().numpy()

                        # Scale back to original size
                        x1 = int(x1 * lane["w"] / lane["resize_w"])
                        y1 = int(y1 * lane["h"] / lane["resize_h"])
                        x2 = int(x2 * lane["w"] / lane["resize_w"])
                        y2 = int(y2 * lane["h"] / lane["resize_h"])

                        vehicles.append([x1, y1, x2, y2])

            lane["last_detection"] = vehicles
        else:
            vehicles = lane["last_detection"]

        # ---------- COUNT VEHICLES ----------
        lane_count = sum(
            1 for box in vehicles
            if point_in_poly(bbox_centroid(box), lane["ROI"])
        )

        # ---------- GREEN TIME LOGIC ----------
        now = time.time()
        if now - lane["last_cycle_time"] >= CYCLE_INTERVAL:
            lane["last_cycle_time"] = now
            proportion = min(lane_count / 10, 1)
            lane["green_time"] = int(G_MIN + proportion * CYCLE_BUDGET)
            lane["green_time"] = max(G_MIN, min(G_MAX, lane["green_time"]))

        # ---------- DRAW ----------
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

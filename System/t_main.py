import cv2
import numpy as np
import time
import serial

from t_detector import VehicleDetector
from t_utils import compute_green_time,count_vehicles_in_roi,send_to_esp32

MODEL_PATH = "yolo11s.pt"
VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

G_MIN = 5
G_MAX = 45
YELLOW_TIME = 2
DISPLAY_W = 800
DISPLAY_H = 450
WINDOW_POSITIONS = {
    "EAST":  (0, 0),                          # top-left
    "SOUTH": (DISPLAY_W, 0),                  # top-right
    "NORTH":  (0, DISPLAY_H+100),                  # bottom-left
    "WEST": (DISPLAY_W, DISPLAY_H+100),          # bottom-right
}
ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
time.sleep(2)  # ESP32 reset time
print("ESP32 connected")

LANES = [
    {
        "name": "EAST",
        "video": "/home/sumankhatri/Videos/c-lane.mp4",
        "ROI": np.array([(93,688),(1262,701),(1273,392),(988,205),(502,209),(92,686)])
    },
    {
        "name": "SOUTH",
        "video": "/home/sumankhatri/Videos/B_lane2.mp4",
        "ROI": np.array([(3, 419), (1, 712), (1275, 695),
                         (1273, 489), (1103, 174), (441, 144)])
    },
    {
        "name": "WEST",
        "video": "/home/sumankhatri/Videos/B_lane3.mp4",
        "ROI": np.array([(3, 419), (1, 712), (1275, 695),
                         (1273, 489), (1103, 174), (441, 144)])
    },
    {
        "name": "NORTH",
        "video": "/home/sumankhatri/Videos/B_lane4.mp4",
        "ROI": np.array([(3, 419), (1, 712), (1275, 695),
                         (1273, 489), (1103, 174), (441, 144)])
    }
]

# ---------------- INIT ----------------
detector = VehicleDetector(MODEL_PATH, VEHICLE_CLASSES)

for lane in LANES:
    lane["cap"] = cv2.VideoCapture(lane["video"])
    lane["green_time"] = G_MIN
    lane["green_start"] = 0
    lane["last_frame"] = None

current_lane_idx = 0
is_yellow_phase = False
yellow_start_time = 0
prev_lane_idx = None
LANES[current_lane_idx]["green_start"] = time.time()

print("Traffic system started")
send_to_esp32(
    ser,
    LANES[current_lane_idx]["name"][0],
    "G",
    LANES[current_lane_idx]["green_time"]
)

# ---------------- MAIN LOOP ----------------
while True:
    now = time.time()
    current_lane = LANES[current_lane_idx]

    # ---- SWITCH LOGIC ----
    if not is_yellow_phase:
        # GREEN phase running
        if now - current_lane["green_start"] >= current_lane["green_time"]:
            # Enter YELLOW phase
            is_yellow_phase = True
            yellow_start_time = now
            prev_lane_idx = current_lane_idx
            # CURRENT lane: GREEN → YELLOW
            send_to_esp32(
                ser,
                LANES[prev_lane_idx]["name"][0],
                "Y",
                YELLOW_TIME
            )

            # NEXT lane: RED → YELLOW
            next_idx = (prev_lane_idx + 1) % len(LANES)
            send_to_esp32(
                ser,
                LANES[next_idx]["name"][0],
                "Y",
                YELLOW_TIME
            )

    else:
        # YELLOW phase running
        if now - yellow_start_time >= YELLOW_TIME:
            # Exit YELLOW → switch lane
            is_yellow_phase = False
            current_lane_idx = (current_lane_idx + 1) % len(LANES)
            next_lane = LANES[current_lane_idx]

            if next_lane["last_frame"] is not None:
                count = count_vehicles_in_roi(
                    detector,
                    next_lane["last_frame"],
                    next_lane["ROI"]
                )
            else:
                count = 0

            next_lane["green_time"] = compute_green_time(count, G_MIN, G_MAX)
            next_lane["green_start"] = time.time()
            # PREVIOUS lane: YELLOW → RED
            send_to_esp32(
                ser,
                LANES[prev_lane_idx]["name"][0],
                "R",
                0
            )

            # NEW lane: YELLOW → GREEN
            send_to_esp32(
                ser,
                next_lane["name"][0],
                "G",
                next_lane["green_time"]
            )


            print(
                f"{next_lane['name']} GREEN | "
                f"Vehicles: {count} | "
                f"Time: {next_lane['green_time']}s"
            )

    # ---- DISPLAY ----
    for idx, lane in enumerate(LANES):
        ret, frame = lane["cap"].read()
        if not ret:
            lane["cap"].set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = lane["cap"].read()
            if not ret:
                continue

        lane["last_frame"] = frame

        cv2.polylines(frame, [lane["ROI"]], True, (255, 255, 0), 2)

        if is_yellow_phase:
        # During yellow: current & next lanes are yellow
            next_idx = (prev_lane_idx + 1) % len(LANES)

            if idx == prev_lane_idx or idx == next_idx:
                cv2.putText(
                    frame,
                    f"{lane['name']} YELLOW",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    f"{lane['name']} RED",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

        else:
            # Normal GREEN / RED phase
            if idx == current_lane_idx:
                remaining = max(
                    0,
                    int(lane["green_time"] - (now - lane["green_start"]))
                )
                cv2.putText(
                    frame,
                    f"{lane['name']} GREEN {remaining}s",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    f"{lane['name']} RED",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )


        display_frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
        cv2.imshow(lane["name"], display_frame)
        x, y = WINDOW_POSITIONS[lane["name"]]
        cv2.moveWindow(lane["name"], x, y)

    # ---- STOP SYSTEM ----
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------- CLEANUP ----------------
for lane in LANES:
    lane["cap"].release()

cv2.destroyAllWindows()

# Turn all ESP32 LEDs to RED
for i in range(4):
    send_to_esp32(ser, LANES[i]["name"][0], "R", 0)

ser.close()

print("Traffic system stopped")

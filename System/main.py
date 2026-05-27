import cv2
import numpy as np
import time
import requests

from detector import VehicleDetector
from utils import count_vehicles_in_roi

MODEL_PATH = "best.pt"
VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

G_MIN = 5
YELLOW_TIME = 2
DISPLAY_W = 800
DISPLAY_H = 450

WINDOW_POSITIONS = {
    "EAST":  (0, 0),
    "SOUTH": (DISPLAY_W, 0),
    "NORTH": (0, DISPLAY_H+100),
    "WEST":  (DISPLAY_W, DISPLAY_H+100),
}

# ESP32 WiFi IP
ESP32_IP = "192.168.4.1"
print("Using ESP32 WiFi at", ESP32_IP)

# ---------------- SEND FUNCTION (WiFi) ----------------
def send_to_esp32(lane, color, t):
    try:
        url = f"http://{ESP32_IP}/set"
        params = {
            "lane": lane,
            "color": color,
            "time": t
        }
        response = requests.get(url, params=params, timeout=2)
        print(f"Sent: LANE {lane} {color} {t} | ESP32: {response.text}")
    except Exception as e:
        print("ESP32 send error:", e)

# ---------------- LANES CONFIG ----------------
LANES = [
    {
        "name": "EAST",
        "video": "/home/sumankhatri/Videos/el.mp4",
        "ROI": np.array([(5,531),(415,525),(496,137),(218,145),(5,258),(4,524)])
    },
    {
        "name": "SOUTH",
        "video": "/home/sumankhatri/Videos/sl.mp4",
        "ROI": np.array([(5,402),(2,711),(1275,711),(1262,413),(1143,223),(292,224),(13,398)])
    },
    {
        "name": "WEST",
        "video": "/home/sumankhatri/Videos/wl.mp4",
        "ROI": np.array([(89,533),(306,175),(476,174),(670,528),(95,533)])
    },
    {
        "name": "NORTH",
        "video": "/home/sumankhatri/Videos/nl.mp4",
        "ROI": np.array([(646,525),(519,146),(287,154),(4,447),(6,528),(640,530)])
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

# Initial GREEN signal
send_to_esp32(
    LANES[current_lane_idx]["name"][0],
    "G",
    LANES[current_lane_idx]["green_time"]
)



start = time.time()
lane_counts = [0] * len(LANES)
green_time_start = time.time()
    
while True:
    now = time.time()
    process_interval = 5

    if now - start >= process_interval:

        for i in range(len(LANES)):
            frame = LANES[i]["last_frame"]

            if frame is not None:
                lane_counts[i] = count_vehicles_in_roi(
                    detector,
                    frame,
                    LANES[i]["ROI"]
                )
            else:
                lane_counts[i] = 0

        print("Vehicle counts:", lane_counts)

        # 🔑 RESET TIMER
        start = now

        if lane_counts[current_lane_idx] == 0 and now - green_time_start >= G_MIN:
            switch = True
        elif now - green_time_start >= G_MIN and max(lane_counts) >= lane_counts[current_lane_idx] * 1.3:
            switch = True
        else:
            switch = False
        
        if switch:
            green_time_start = time.time()
            # Enter YELLOW phase
            is_yellow_phase = True
            yellow_start_time = now
            prev_lane_idx = current_lane_idx
            # Current lane → YELLOW
            send_to_esp32(LANES[prev_lane_idx]["name"][0], "Y", YELLOW_TIME)
            # Next lane → YELLOW #technically the lane were gonna go green
            next_idx = (prev_lane_idx + 1) % len(LANES)
            send_to_esp32(LANES[next_idx]["name"][0], "Y", YELLOW_TIME)


        if is_yellow_phase and now - yellow_start_time >= YELLOW_TIME:
            is_yellow_phase = False
            current_lane_idx = (current_lane_idx + 1) % len(LANES)

            green_time_start = now
            next_lane = LANES[current_lane_idx]
            send_to_esp32(LANES[current_lane_idx]["name"][0], "G", 5)
            send_to_esp32(LANES[prev_lane_idx]["name"][0], "R", 0)




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
            next_idx = (prev_lane_idx + 1) % len(LANES)

            if idx == prev_lane_idx or idx == next_idx:
                cv2.putText(frame, f"{lane['name']} YELLOW",
                            (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 255), 2)
            else:
                cv2.putText(frame, f"{lane['name']} RED",
                            (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)
        else:
            if idx == current_lane_idx:
                cv2.putText(frame, f"{lane['name']} GREEN",
                            (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"{lane['name']} RED",
                            (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)

        display_frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
        cv2.imshow(lane["name"], display_frame)

        x, y = WINDOW_POSITIONS[lane["name"]]
        cv2.moveWindow(lane["name"], x, y)

    # ---- STOP ----
    delay = 33  # ~30 FPS (good enough for most videos)
    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

# ---------------- CLEANUP ----------------
for lane in LANES:
    lane["cap"].release()

cv2.destroyAllWindows()

# Turn all signals RED at end
for lane in LANES:
    send_to_esp32(lane["name"][0], "R", 0)

print("Traffic system stopped")
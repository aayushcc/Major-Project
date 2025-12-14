import serial
import time

# ================= CONFIGURATION =================
PORT = "/dev/ttyUSB0"   # Change if your ESP32 uses a different port
BAUD = 115200

# Example vehicle count per lane (replace with YOLO output later)
vehicle_count = [10, 5, 3, 8]

# Timing limits
MIN_GREEN = 5
MAX_GREEN = 20
YELLOW_TIME = 2
RED_GAP = 0.5  # 0.5 seconds gap between lanes

# Lane mapping
lane_names = ["North", "East", "South", "West"]

# ================== FUNCTIONS ====================
def calculate_green(count):
    """Calculate green light time based on vehicle count"""
    return min(MAX_GREEN, max(MIN_GREEN, count * 2))

def send_lane(ser, lane, green_time):
    """Send lane + green time command to ESP32"""
    cmd = f"LANE {lane} {green_time}\n"
    ser.write(cmd.encode())
    print(f"Sent: {cmd.strip()}")

def read_esp32_feedback(ser, timeout=0.1):
    """Read lines from ESP32 for debugging"""
    start = time.time()
    while time.time() - start < timeout:
        while ser.in_waiting:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    print("ESP32:", line)
            except:
                pass  # ignore any other decode errors


# ================== MAIN =========================
def main():
    # Connect to ESP32
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # Wait for ESP32 to initialize
    print("Connected to ESP32.")

    # Initial feedback
    read_esp32_feedback(ser, 1)

    # Main traffic sequence
    for lane in range(4):
        count = vehicle_count[lane]
        green_time = calculate_green(count)

        print(f"\nLane {lane_names[lane]}: vehicle count = {count}, green time = {green_time}s")
        send_lane(ser, lane, green_time)

        # Wait for lane to finish (green + yellow + gap)
        wait_time = green_time + YELLOW_TIME + RED_GAP
        for t in range(int(wait_time)):
            read_esp32_feedback(ser, 0.1)
            time.sleep(1)

    # Close serial connection
    ser.close()
    print("Traffic sequence complete.")

# ================== RUN ==========================
if __name__ == "__main__":
    main()

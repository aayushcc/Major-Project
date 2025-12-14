import serial
import time

ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
time.sleep(2)

def send_lane(lane, green_time):
    cmd = f"LANE {lane} {green_time}\n"
    ser.write(cmd.encode())
    print("Sent:", cmd.strip())

# Example (from AI detection)
send_lane(0, 15)  # North → 15s
time.sleep(20)

send_lane(1, 10)  # East → 10s

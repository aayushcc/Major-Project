import cv2
import serial
import time


def point_in_poly(pt, poly):
    return cv2.pointPolygonTest(poly, pt, False) >= 0


def bbox_centroid(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


# def compute_green_time(vehicle_count, g_min, g_max):
#     ratio = min(vehicle_count / 10.0, 1.0)
#     return int(g_min + ratio * (g_max - g_min))


def compute_green_time(vc, g_min, g_max):
    if vc <= 5:
        ratio = vc / 5
    elif vc <= 20:
        ratio = 0.6 + 0.4 * (vc - 5) / 15
    else:
        ratio = 1.0
    return int(g_min + ratio * (g_max - g_min)), ratio


def count_vehicles_in_roi(detector, frame, roi):
    boxes = detector.detect(frame)
    count = 0
    for box in boxes:
        if point_in_poly(bbox_centroid(box), roi):
            count += 1
    return count


def send_to_esp32(ser, lane, color, duration):
    if ser is None or not ser.is_open:
        return

    msg = f"{lane},{color},{duration}\n"
    ser.write(msg.encode())
    ser.flush()

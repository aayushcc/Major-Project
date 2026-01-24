import cv2
import numpy as np

points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point added: ({x}, {y})")

cap = cv2.VideoCapture("/home/sumankhatri/Videos/B_lane1.mp4")

ret, frame = cap.read()
cap.release()

cv2.imshow("Click ROI points (Press ENTER when done)", frame)
cv2.setMouseCallback("Click ROI points (Press ENTER when done)", mouse_callback)

while True:
    temp = frame.copy()
    if len(points) > 1:
        cv2.polylines(temp, [np.array(points)], False, (0, 255, 0), 2)
    for p in points:
        cv2.circle(temp, p, 5, (0, 0, 255), -1)

    cv2.imshow("Click ROI points (Press ENTER when done)", temp)
    key = cv2.waitKey(1)

    if key == 13:  # ENTER key
        break

cv2.destroyAllWindows()

print("\nFinal ROI:")
print("ROI =", np.array(points))

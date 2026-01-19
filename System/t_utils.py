import cv2

def point_in_poly(pt, poly):
    return cv2.pointPolygonTest(poly, pt, False) >= 0

def bbox_centroid(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def read_latest_frame(cap, max_drop=8):
    """
    Always returns the most recent frame.
    Drops older frames to prevent lag buildup.
    """
    frame = None
    for _ in range(max_drop):
        ret, f = cap.read()
        if not ret:
            break
        frame = f
    return frame

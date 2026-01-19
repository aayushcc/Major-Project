import torch
from ultralytics import YOLO
import cv2

class VehicleDetector:
    def __init__(self, model_path, vehicle_classes, conf=0.4):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Using device: {self.device}")

        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.vehicle_classes = vehicle_classes
        self.conf = conf

    def detect(self, frame, resize_shape, original_shape):
        """
        Runs YOLO on a resized frame and
        returns bounding boxes scaled to original size.
        """
        resized = cv2.resize(frame, resize_shape)

        results = self.model(
            resized,
            conf=self.conf,
            device=self.device,
            half=(self.device == "cuda"),
            verbose=False
        )

        boxes_out = []
        r = results[0]

        if r.boxes is None:
            return boxes_out

        ow, oh = original_shape
        rw, rh = resize_shape

        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            if self.model.names[int(cls)] in self.vehicle_classes:
                x1, y1, x2, y2 = box.cpu().numpy()

                # scale back
                x1 = int(x1 * ow / rw)
                y1 = int(y1 * oh / rh)
                x2 = int(x2 * ow / rw)
                y2 = int(y2 * oh / rh)

                boxes_out.append([x1, y1, x2, y2])

        return boxes_out

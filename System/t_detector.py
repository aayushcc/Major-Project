import torch
from ultralytics import YOLO


class VehicleDetector:
    def __init__(self, model_path, vehicle_classes, conf=0.4, imgsz=512):
        """
        YOLO handles resizing + letterboxing internally.
        """

        self.device = 0 if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model = YOLO(model_path)

        self.conf = conf
        self.imgsz = imgsz

        # map class names -> class ids once
        self.vehicle_class_ids = [
            k for k, v in self.model.names.items()
            if v in vehicle_classes
        ]

        if not self.vehicle_class_ids:
            raise ValueError("No valid vehicle classes found in model")

    def detect(self, frame):
        """
        Runs YOLO on the original frame.
        Returns bounding boxes in original image coordinates.
        """

        results = self.model(
            frame,
            conf=self.conf,
            device=self.device,
            classes=self.vehicle_class_ids,
            imgsz=self.imgsz,
            half=(self.device != "cpu"),
            verbose=False
        )

        boxes_out = []
        r = results[0]

        if r.boxes is None:
            return boxes_out

        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            boxes_out.append([x1, y1, x2, y2])

        return boxes_out

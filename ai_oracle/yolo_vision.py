from ultralytics import YOLO
from rich.console import Console

console = Console()


class YoloVision:
    """Classifies objects in an image using a YOLO model."""

    def __init__(self, model_name="yolo11n.pt", confidence=0.4):
        """
        Args:
            model_name: YOLO model weight file. 'yolo11n.pt' is the smallest/fastest.
                        Other options: 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt'
            confidence: Minimum confidence threshold for detections.
        """
        self.confidence = confidence
        console.print(f"[cyan]Loading YOLO model: {model_name}...[/cyan]")
        self.model = YOLO(model_name)
        console.print("[green]YOLO model loaded![/green]")

    def classify(self, frame):
        """
        Run YOLO object detection on a single OpenCV BGR frame.

        Args:
            frame: A numpy BGR image (OpenCV format).

        Returns:
            A list of dicts, each containing:
                - 'label': class name (e.g. 'person', 'cat', 'dog')
                - 'confidence': float 0.0-1.0
                - 'box': (x1, y1, x2, y2) bounding box coordinates
        """
        results = self.model(frame, conf=self.confidence, verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    'label': label,
                    'confidence': conf,
                    'box': (int(x1), int(y1), int(x2), int(y2))
                })

        return detections

import cv2
import os
import tempfile
from rich.console import Console

console = Console()


class ImageProcessor:
    """Detects people in an image using face detection and crops to include them."""

    def __init__(self, padding_factor=0.8, scale_factor=0.75):
        """
        Args:
            padding_factor: Multiplier around detected face to include upper body.
            scale_factor: Scale factor to reduce image quality/size (0.0 - 1.0).
        """
        self.padding_factor = padding_factor
        self.scale_factor = scale_factor

        # Load Haar cascade classifiers for face detection
        self.face_cascade_frontal = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.face_cascade_profile = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )

    def _detect_faces(self, gray_image):
        """Try frontal detection first, then profile if nothing found."""
        faces = self.face_cascade_frontal.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        if len(faces) == 0:
            # Try profile (side) face detection
            faces = self.face_cascade_profile.detectMultiScale(
                gray_image,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
        return faces

    def process(self, image_path: str) -> str:
        """
        Detect people (via faces) in the image and crop to include them.
        Returns the path to the processed image.
        If no faces are detected, returns the original image (downscaled).
        """
        console.print("[cyan]Processing image: detecting people...[/cyan]")

        image = cv2.imread(image_path)
        if image is None:
            console.print("[yellow]Warning: Could not read image for processing. Using original.[/yellow]")
            return image_path

        original_h, original_w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self._detect_faces(gray)

        if len(faces) > 0:
            console.print(f"[green]Detected {len(faces)} face(s)![/green]")

            # For each face, expand the region to include the upper body
            regions = []
            for (fx, fy, fw, fh) in faces:
                pad_x = int(fw * self.padding_factor)
                pad_y = int(fh * self.padding_factor)

                # Tight crop: small upward, moderate downward for shoulders
                region_x1 = max(0, fx - pad_x)
                region_y1 = max(0, fy - int(pad_y * 0.4))
                region_x2 = min(original_w, fx + fw + pad_x)
                region_y2 = min(original_h, fy + fh + int(pad_y * 1.2))
                regions.append((region_x1, region_y1, region_x2, region_y2))

            # Merge all regions into one bounding box
            min_x = min(r[0] for r in regions)
            min_y = min(r[1] for r in regions)
            max_x = max(r[2] for r in regions)
            max_y = max(r[3] for r in regions)

            cropped = image[min_y:max_y, min_x:max_x]
        else:
            console.print("[yellow]No faces detected. Using full image.[/yellow]")
            cropped = image

        # Downscale the image to reduce size for faster LLM processing
        if self.scale_factor < 1.0:
            new_w = int(cropped.shape[1] * self.scale_factor)
            new_h = int(cropped.shape[0] * self.scale_factor)
            if new_w > 0 and new_h > 0:
                cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Save processed image to a temp file
        temp_dir = tempfile.gettempdir()
        processed_path = os.path.join(temp_dir, "ai_oracle_processed.jpg")
        cv2.imwrite(processed_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 80])
        console.print("[green]Image processing complete.[/green]")

        return processed_path


import cv2
import tempfile
import time
import os
import typer
from rich.console import Console

console = Console()

class CameraClient:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index

    def capture_image(self) -> str:
        """Capture an image from the default camera and return the path to the temporary image file."""
        console.print("[cyan]Accessing the camera...[/cyan]")
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            console.print("[bold red]Error:[/bold red] Could not open the camera. Please ensure camera permissions are granted to your Terminal.")
            raise typer.Exit(code=1)

        for _ in range(10):
            cap.read()
            time.sleep(0.05)
            
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            console.print("[bold red]Error:[/bold red] Failed to capture image from the camera.")
            raise typer.Exit(code=1)

        temp_dir = tempfile.gettempdir()
        image_path = os.path.join(temp_dir, "ai_oracle_capture.jpg")
        cv2.imwrite(image_path, frame)
        console.print(f"[green]Image captured successfully![/green]")
        
        return image_path

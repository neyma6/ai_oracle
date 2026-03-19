import os
# Fix macOS 26+ compatibility issue with older tkinter/Tcl setups
os.environ["SYSTEM_VERSION_COMPAT"] = "0"
import typer
from rich.console import Console
from ai_oracle.tplink_camera import TpLinkCamera
from ai_oracle.motion_detection import MotionDetector
from ai_oracle.yolo_vision import YoloVision
from ai_oracle.ai_camera_processing import AiCameraProcessing

app = typer.Typer(help="Ai_Oracle: Live Camera Stream", no_args_is_help=True)
console = Console()

@app.command()
def analyze():
    """Open a live video feed from the TP-Link camera with motion detection, YOLO classification, and AI analysis."""
    from ai_oracle.ui_client import VisionApp

    camera_client = TpLinkCamera()
    motion_detector = MotionDetector()
    yolo_vision = YoloVision()
    ai_processor = AiCameraProcessing()
    console.print("[cyan]Opening Live Stream Interface...[/cyan]")
    
    # Launch GUI
    app_ui = VisionApp(camera_client=camera_client, motion_detector=motion_detector, yolo_vision=yolo_vision, ai_processor=ai_processor)
    app_ui.mainloop()

if __name__ == "__main__":
    app()

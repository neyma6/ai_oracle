import os
import cv2
import tempfile
import urllib.parse
from rich.console import Console

console = Console()

class TpLinkCamera:
    """Connects to a TP-Link (Tapo/Kasa) camera via RTSP and provides the video stream."""

    def __init__(self, ip_address=None, video_file=None):
        self.video_file = video_file
        self.username, self.password, file_ip = self._load_credentials()
        # Default to IP from file (if any), then provided arg, then env var, then default 192.168.1.100
        self.ip_address = file_ip or ip_address or os.environ.get("TPLINK_CAMERA_IP", "192.168.1.100")

    def _load_credentials(self):
        """Load username, password, and optional IP from ~/.ai_oracle/camera"""
        camera_creds_path = os.path.expanduser("~/.ai_oracle/camera")
        if os.path.exists(camera_creds_path):
            with open(camera_creds_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                if len(lines) >= 3:
                    return lines[0], lines[1], lines[2]
                elif len(lines) == 2:
                    return lines[0], lines[1], None
        
        console.print(f"[yellow]Warning: Camera credentials not found in {camera_creds_path}[/yellow]")
        return None, None, None

    def get_rtsp_url(self, high_res=True) -> str:
        """Construct the RTSP URL based on TP-Link's standard format."""
        if not self.username or not self.password:
            raise ValueError("Camera credentials missing. Please add user/pass to ~/.ai_oracle/camera")

        # URL encode credentials in case they have special symbols like '@' or '#'
        user_enc = urllib.parse.quote(self.username)
        pass_enc = urllib.parse.quote(self.password)
        
        # TP-link Tapo/Kasa cameras typically use stream1 for HD, stream2 for SD (360p)
        stream_path = "stream1" if high_res else "stream2"
        
        return f"rtsp://{user_enc}:{pass_enc}@{self.ip_address}:554/{stream_path}"

    def get_stream(self):
        """
        Connects to the RTSP URL and yields frames as a live stream.
        Can be iterated over:
            for frame in camera.get_stream():
                ...
        """
        cap = None
        fps = 30
        is_local_file = False
        
        if self.video_file:
            rtsp_url = self.video_file
            console.print(f"[cyan]Playing local video file {self.video_file}...[/cyan]")
            # Do not use FFMPEG backend flag exclusively for local files
            cap = cv2.VideoCapture(rtsp_url)
            is_local_file = True
            file_fps = cap.get(cv2.CAP_PROP_FPS)
            if file_fps > 0:
                fps = file_fps
        else:
            rtsp_url = self.get_rtsp_url()
            console.print(f"[cyan]Connecting to TP-Link camera stream at {self.ip_address}...[/cyan]")
            # We use cv2.CAP_FFMPEG for network streams to reduce latency in OpenCV
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            
        if not cap.isOpened():
            raise RuntimeError(f"Could not open TP-Link camera stream or file at {rtsp_url}")

        import time
        frame_delay = 1.0 / fps if is_local_file else 0

        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    console.print("[yellow]Warning: Camera stream disconnected or video ended.[/yellow]")
                    break
                yield frame
                
                if is_local_file:
                    elapsed = time.time() - start_time
                    sleep_time = frame_delay - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        finally:
            cap.release()
            
    def capture_image(self) -> str:
        """
        Capture a single snapshot frame from the stream (compatible with the current UI).
        Returns the path to the saved temporary image file.
        """
        cap = cv2.VideoCapture(self.get_rtsp_url(), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open TP-Link camera at {self.ip_address}")

        try:
            # Grab a few frames first so the camera sensor adjusts its white balance/exposure
            for _ in range(10):
                cap.grab()
                
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read snapshot frame from TP-Link camera")

            temp_dir = tempfile.gettempdir()
            image_path = os.path.join(temp_dir, "ai_oracle_tplink_snapshot.jpg")
            
            # Save it
            cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            console.print("[green]TP-Link snapshot captured successfully![/green]")
            
            return image_path
            
        finally:
            cap.release()

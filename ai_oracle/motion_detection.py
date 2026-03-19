import cv2
import time
import numpy as np

class MotionDetector:
    def __init__(self, min_area=2000, cooldown_seconds=2):
        self.min_area = min_area
        self.cooldown_seconds = cooldown_seconds
        self.last_motion_time = 0
        
        # Use OpenCV's official Background Subtractor algorithm.
        # This solves the "Picasso-mask" problem by learning the environment precisely and 
        # isolating exactly the pixels that are foreign/moving.
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    def detect(self, frame):
        """
        Calculates background subtraction.
        Returns a cropped snapshot over a solid black (or white) canvas.
        """
        # Apply the background subtractor
        fgmask = self.fgbg.apply(frame)

        # MOG2 labels pure moving foreground as 255. Shadows are labeled as 127.
        # We threshold > 200 to drop shadows and keep only the true moving object.
        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up sensor noise and merge body-parts into one solid mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Heavily dilate so the internal parts of the object (if they happen to match background colors)
        # expand to fill the entire silhouette solidly.
        thresh = cv2.dilate(thresh, None, iterations=6)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        min_x = frame.shape[1]
        min_y = frame.shape[0]
        max_x = 0
        max_y = 0

        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue

            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

        if motion_detected:
            current_time = time.time()
            if current_time - self.last_motion_time > self.cooldown_seconds:
                self.last_motion_time = current_time
                
                # Check average color of the moving pixels
                mean_color = cv2.mean(frame, mask=thresh)
                brightness = (mean_color[0] + mean_color[1] + mean_color[2]) / 3.0
                
                # Set background to White if the moving object is extremely dark, else Black
                bg_color = (255, 255, 255) if brightness < 30 else (0, 0, 0)
                bg_img = np.full_like(frame, bg_color)
                
                # Combine the actual moving pixels with the solid background
                mask_bool = (thresh > 0)[..., np.newaxis]
                result_frame = np.where(mask_bool, frame, bg_img)
                
                # Crop to the tightly bounded region to generate the final snapshot
                pad = 20
                crop_y1 = max(0, min_y - pad)
                crop_y2 = min(frame.shape[0], max_y + pad)
                crop_x1 = max(0, min_x - pad)
                crop_x2 = min(frame.shape[1], max_x + pad)
                
                return result_frame[crop_y1:crop_y2, crop_x1:crop_x2]
                
        return None


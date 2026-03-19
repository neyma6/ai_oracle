import ollama
import cv2
import base64
import os
import re
import threading
import time


class AiCameraProcessing:
    """Sends a snapshot image + YOLO classification to an LLM for a natural language description."""

    def __init__(self, model="qwen3.5:4b", cooldown_seconds=10):
        """
        Args:
            model: Ollama model name to use for vision analysis.
            cooldown_seconds: Ignored, replaced by queue system.
        """
        self.model = model

    def _extract_response(self, raw_content):
        """Extract useful text from the LLM response, handling <think> tags.
        
        Qwen thinking models wrap their entire reasoning in <think>...</think> 
        and put the final answer outside. If nothing is outside the tags, 
        we fall back to extracting the last sentence from inside the tags.
        """
        if not raw_content:
            return None

        # Try to get content outside <think> tags first
        outside = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
        if outside:
            return outside

        # Fallback: extract content from inside the think tags
        think_match = re.search(r'<think>(.*?)</think>', raw_content, flags=re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()
            # Take the last non-empty line as the "conclusion"
            lines = [l.strip() for l in think_content.split('\n') if l.strip()]
            if lines:
                return lines[-1]

        # Last resort: return the raw content stripped of tags
        cleaned = re.sub(r'</?think>', '', raw_content).strip()
        return cleaned if cleaned else None

    def analyze(self, frame, classifications, similar_context=None):
        """
        Resize the frame, save as temp file, and send it to the LLM.
        Processed sequentially via UI queue.

        Args:
            frame: Original BGR OpenCV frame (full resolution).
            classifications: List of dicts with 'label' and 'confidence' keys.
            similar_context: Optional list of strings (descriptions) from past similar events.

        Returns:
            A string with the LLM's one-sentence description, or None if skipped.
        """
        # Resize to 640px wide — enough detail for the VLM
        height, width = frame.shape[:2]
        target_width = 640
        scale = target_width / width
        resized = cv2.resize(frame, (target_width, int(height * scale)))

        # Encode directly to JPEG bytes — no temp file needed, avoids file race conditions
        success, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if not success:
            print("[AI Camera] Failed to encode frame to JPEG")
            return None
        image_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')

        try:
            # Build classification context
            if classifications:
                class_str = ", ".join([f"{d['label']} ({d['confidence']:.0%})" for d in classifications])
            else:
                class_str = "unknown object"

            prompt = f"Describe the detected objects in one sentence. Detected objects: {class_str}."

            messages = []
            if similar_context:
                # Inject past description as assistant history to prime the model
                messages.append({
                    'role': 'user',
                    'content': 'Describe the previous image.'
                })
                messages.append({
                    'role': 'assistant',
                    'content': similar_context[0]
                })
                prompt += " Note: if this appears to be the same objects, explicitly say so."

            messages.append({
                'role': 'user',
                'content': prompt,
                'images': [image_b64]
            })

            # qwen3.5:4b's thinking mode silently produces empty output sometimes.
            # Disabling it with think=False ensures we always get raw text output.
            max_retries = 3
            content = None

            for attempt in range(max_retries):
                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        'num_predict': 300,
                        'temperature': 0.4,
                    },
                    think=False,
                    keep_alive='10m',
                )

                if hasattr(response, 'message'):
                    raw = getattr(response.message, 'content', '') or ''
                else:
                    msg = response.get('message', {})
                    raw = msg.get('content', '') or ''

                print(f"[AI Camera] LLM raw response (Attempt {attempt+1}): >>>{raw}<<<")

                content = self._extract_response(raw)
                if content and content.strip():
                    break

                print(f"[AI Camera] LLM output was empty, retrying... ({attempt+1}/{max_retries})")
                time.sleep(2.0)

            print(f"[AI Camera] Extracted: '{content}'")
            return content

        except Exception as e:
            print(f"[AI Camera] LLM Error: {e}")
            return None


import ollama
import cv2
import tempfile
import os


class AiCameraProcessing:
    """Sends a snapshot image + YOLO classification to an LLM for a natural language description."""

    def __init__(self, model="qwen3.5:4b"):
        """
        Args:
            model: Ollama model name to use for vision analysis.
        """
        self.model = model

    def analyze(self, frame, classifications):
        """
        Resize the frame to 50%, save it as a temp file, and send it to the LLM
        along with the YOLO classification labels.

        Args:
            frame: Original BGR OpenCV frame (full resolution).
            classifications: List of dicts with 'label' and 'confidence' keys.

        Returns:
            A string with the LLM's one-sentence description.
        """
        # Resize to 50% of original
        height, width = frame.shape[:2]
        resized = cv2.resize(frame, (width // 2, height // 2))

        # Save to a temporary file for Ollama
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, resized)

            # Build classification context
            if classifications:
                class_str = ", ".join([f"{d['label']} ({d['confidence']:.0%})" for d in classifications])
            else:
                class_str = "unknown object"

            prompt = f"/no_think What is on the image? Use the provided classification: {class_str}. Answer only with one sentence."

            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [tmp_path]
                }],
                options={'num_predict': 512},
            )

            # Handle both object (new SDK) and dict (old SDK) response formats
            if hasattr(response, 'message'):
                content = getattr(response.message, 'content', '') or ''
            else:
                msg = response.get('message', {})
                content = msg.get('content', '') or ''

            # Strip <think>...</think> tags that qwen3.5 wraps its reasoning in
            import re
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            
            print(f"[AI Camera] LLM response: '{content}'")
            return content if content else "[No answer after thinking]"

        except Exception as e:
            return f"[LLM Error: {e}]"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass

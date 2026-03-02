import ollama

class AIClient:
    def __init__(self, model: str = "llama3.2"):
        self.model = model

    def stream_chat(self, prompt: str, context: list = []):
        
        messages = context + [
            {'role': 'system', 'content': 'You are a helpful software engineer assistant. You only give a short sentence by answer and you provide code snippets if it is needed.'},
            {'role': 'user', 'content': prompt}
        ]
        
        stream = ollama.chat(
            model=self.model,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            yield chunk['message']['content']
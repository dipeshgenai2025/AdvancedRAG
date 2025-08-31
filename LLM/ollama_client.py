# LLM/ollama_client.py

"""
Ollama client for embeddings and LLM.
Thread-safe HTTP calls.
"""

import requests
import json
import logging

class OllamaClient:
    def __init__(self, model: str = "mistral:7b", url: str = "http://localhost:11434"):
        self.model = model
        self.url = url

    def generate_answer(self, prompt: str, context: str = "", max_tokens: int = 512) -> str:
        """Generate answer using Ollama with provided context."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(context, str):
            raise ValueError("context must be a string")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")

        full_prompt = f"""You are an assistant. Use the context to answer the question.

        Context:
        {context}

        Question:
        {prompt}

        Answer:"""

        try:
            response = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "options": {"num_predict": max_tokens},
                    "stream": True
                },
                stream=True
            )

            if response.status_code != 200:
                logging.error(f"Ollama request failed with status {response.status_code}: {response.text}")
                raise RuntimeError(f"Ollama request failed: {response.text}")

            # Collect streamed chunks
            answer_parts = []
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if "response" in data:
                            answer_parts.append(data["response"])
                    except json.JSONDecodeError as e:
                        logging.warning(f"JSON decode error: {e}")
                        continue

            return "".join(answer_parts).strip()
        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            raise RuntimeError(f"Answer generation failed: {e}")

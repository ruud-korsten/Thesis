import openai
import os
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self, model: str = None, temperature: float = 0.2):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature

    def call(self, prompt: str) -> str:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response['choices'][0]['message']['content'].strip()
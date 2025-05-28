import os
import openai
from dotenv import load_dotenv
from data_quality_tool.logging_config import get_logger

logger = get_logger()

load_dotenv()


class LLMClient:
    def __init__(self, model: str = None, temperature: float = 0.2):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        logger.info("Initialized LLMClient with model='%s' and temperature=%.2f", self.model, self.temperature)

    def call(self, prompt: str) -> str:
        logger.debug("Calling LLM with prompt length: %d", len(prompt))
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            result = response['choices'][0]['message']['content'].strip()
            logger.info("LLM response received (%d characters)", len(result))
            return result
        except Exception as e:
            logger.exception("Error calling OpenAI API: %s", str(e))
            raise

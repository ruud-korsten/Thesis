import os
import openai
from dotenv import load_dotenv
from data_quality_tool.config.logging_config import get_logger

logger = get_logger()

load_dotenv()

class LLMClient:
    def __init__(self, model: str = None, temperature: float = 0.2):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature

        if "deepseek" in self.model.lower():
            openai.api_key = os.getenv("DEEPSEEK_API_KEY")
            openai.api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")  # adjust if needed
            logger.info("Initialized LLMClient with DeepSeek model='%s'", self.model)
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            # openai.api_base uses default OpenAI endpoint
            logger.info("Initialized LLMClient with OpenAI model='%s'", self.model)

    def call(self, prompt: str = None, messages: list[dict] = None) -> str:
        if not prompt and not messages:
            raise ValueError("You must provide either a prompt or a messages list.")

        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        logger.debug("Calling LLM with %d messages", len(messages))
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            result = response['choices'][0]['message']['content'].strip()
            logger.info("LLM response received (%d characters)", len(result))
            return result
        except Exception as e:
            logger.exception("Error calling LLM API: %s", str(e))
            raise

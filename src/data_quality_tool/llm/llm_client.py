import os
import time
import random
import openai
import openai.error as oe
from dotenv import load_dotenv
from data_quality_tool.config.logging_config import get_logger

logger = get_logger()
load_dotenv()


class LLMClient:
    """Lightweight wrapper around openai.ChatCompletion with cost tracking and robust retries."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.2,
        max_attempts: int = 5,
        backoff_base: float = 1.0,
        request_timeout: int | None = 60,
    ):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.request_timeout = request_timeout

        # --- endpoint / key selection ---------------------------------------------------------
        if "deepseek" in self.model.lower():
            openai.api_key = os.getenv("DEEPSEEK_API_KEY")
            openai.api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
            logger.info("Initialized LLMClient with DeepSeek model='%s'", self.model)

        elif "mistral" in self.model.lower():
            openai.api_base = "https://blenddata-ai-vision-pro-resource.services.ai.azure.com/models"
            openai.api_key = os.getenv("MISTRAL_API_KEY")

        elif "grok" in self.model.lower():
            openai.api_base = "https://blenddata-ai-vision-pro-resource.services.ai.azure.com/models"
            openai.api_key = os.getenv("GROK_API_KEY")

        else:  # OpenAI default
            openai.api_key = os.getenv("OPENAI_API_KEY")
            openai.api_base = "https://api.openai.com/v1"
            logger.info("Initialized LLMClient with OpenAI model='%s'", self.model)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def call(self, prompt: str | None = None, messages: list[dict] | None = None) -> tuple[str, dict]:
        """Send a chat completion request with automatic retries on transient 5xx errors."""
        if not prompt and not messages:
            raise ValueError("You must provide either a prompt or a messages list.")

        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        for attempt in range(1, self.max_attempts + 1):
            try:
                logger.debug("Calling LLM (attempt %d/%d, %d messages)",
                             attempt, self.max_attempts, len(messages))
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    request_timeout=self.request_timeout,
                )

                result_text = response["choices"][0]["message"]["content"].strip()
                usage_info = self._log_and_package_usage(response.get("usage", {}))
                return result_text, usage_info

            except oe.APIError as e:
                if self._should_retry(e, attempt):
                    self._sleep_before_retry(e, attempt)
                    continue
                logger.exception("APIError not retryable or max attempts reached: %s", e)
                raise

            except oe.ServiceUnavailableError as e:  # Handles some 5xxs in old SDKs
                if self._should_retry(e, attempt, http_status=503):
                    self._sleep_before_retry(e, attempt)
                    continue
                logger.exception("ServiceUnavailableError, giving up: %s", e)
                raise

            except Exception as e:
                # All other exceptions are considered fatal (network, parsing, etc.)
                logger.exception("Unhandled exception while calling LLM: %s", e)
                raise

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _should_retry(self, err: Exception, attempt: int, http_status: int | None = None) -> bool:
        """Return True if the error is retryable and we have attempts left."""
        status = getattr(err, "http_status", http_status)
        return status in {502, 503, 504} and attempt < self.max_attempts

    def _sleep_before_retry(self, err: Exception, attempt: int) -> None:
        """Exponential back-off with jitter."""
        delay = self.backoff_base * (2 ** (attempt - 1)) + random.random()
        status = getattr(err, "http_status", "n/a")
        logger.warning("Transient %s error (%s). Retrying in %.1f s… (%d/%d)",
                       err.__class__.__name__, status, delay, attempt, self.max_attempts)
        time.sleep(delay)

    def _log_and_package_usage(self, usage: dict) -> dict:
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        cost = self._estimate_cost(prompt_tokens, completion_tokens)

        logger.info("Token usage – prompt: %d, completion: %d, total: %d",
                    prompt_tokens, completion_tokens, total_tokens)
        logger.info("Estimated cost for this call: $%.6f", cost)

        usage.update({
            "estimated_cost": cost,
            "model": self.model,
        })
        return usage

    # (unchanged) -------------------------------------------------------------
    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = {
            "gpt-4o-mini": (0.15, 0.6),
            "gpt-4o": (2.5, 10),
            "mistral-medium-2505-ruud": (0.4, 2),
            "deepseek-chat": (0.27, 1.1),
            "grok-3": (3.0, 15.0),
            "deepseek-reasoner": (0.55, 2.19),
            "grok-3-mini": (0.3, 0.5),
        }
        model_key = next((k for k in pricing if k in self.model.lower()), None)
        if not model_key:
            return 0.0
        in_price, out_price = pricing[model_key]
        return (prompt_tokens / 1_000_000) * in_price + (completion_tokens / 1_000_000) * out_price

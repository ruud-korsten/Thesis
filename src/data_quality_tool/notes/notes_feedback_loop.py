import pandas as pd
from typing import Dict, Any
from data_quality_tool.config.logging_config import get_logger
from data_quality_tool.llm.llm_client import LLMClient

logger = get_logger()


class NoteFeedbackLoop:
    def __init__(self, model: str = None, temperature: float = 1.0, df: pd.DataFrame = None):
        """
        Initialize the feedback loop engine.
        Args:
            model: LLM model name (optional).
            temperature: Sampling temperature for LLM.
            df: The DataFrame to validate against (required for retries).
        """
        self.llm = LLMClient(model=model, temperature=temperature)
        self.df = df
        self.columns = list(df.columns) if df is not None else []

    def fix_failed_note(self, note: str, code: str, error: str) -> tuple[str, dict]:
        """
        Uses the LLM to repair a broken function by providing the note, broken code, and error.
        Returns clean Python code with no markdown or comments.
        """
        column_str = ", ".join(self.columns)

        system_message = {
            "role": "system",
            "content": (
                "You are a Python code repair assistant for data quality checks. "
                "You are given a domain note, a broken function, and the error that occurred when evaluating it. "
                "Your task is to generate a corrected function that satisfies the note and avoids the error."
            )
        }

        user_message = {
            "role": "user",
            "content": f"""
Domain Note:
{note}

Available Columns:
{column_str}

Broken Function:
{code}

Error Message:
{error}

---

Please return a fixed Python function that:
- Takes a pandas DataFrame `df` as input
- Returns a pandas Series of booleans where `True` means the row violates the rule
- Uses only standard pandas (no .apply, no external libraries)
- Uses only the listed columns (exact names)
- Has a descriptive function name in snake_case
- Does not include markdown, comments, or explanation — just the code
""".strip()
        }

        try:
            logger.debug("Sending LLM request for note: %s", note)

            raw_response, usage = self.llm.call(messages=[system_message, user_message])
            logger.debug("LLM raw response:\n%s", raw_response)

            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                cleaned = "\n".join(line for line in lines if not line.strip().startswith("```")).strip()

            logger.debug("LLM cleaned function:\n%s", cleaned)
            return cleaned, usage

        except Exception as e:
            logger.exception("Failed to call LLM during function repair for note: %s — %s", note, str(e))
            raise

    def retry_failed_notes(self, note_results: Dict[str, Dict[str, Any]]
                           ) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        Retry all failed notes.

        Returns
        -------
        repaired   : notes that were fixed (or still failed) with metadata
        usage_dict : {note: usage_stats} for every LLM retry call
        """
        repaired: Dict[str, Dict[str, Any]] = {}
        usage_dict: Dict[str, Dict[str, Any]] = {}

        for note, result in note_results.get("failed", {}).items():
            logger.info("Retrying failed note using LLM: %s", note)

            try:
                fixed_code, usage = self.fix_failed_note(
                    note=note, code=result["code"], error=result["error"]
                )

                usage_dict[note] = usage  #collect usage

                repaired[note] = {
                    "id": result["id"],
                    "function_name": func.__name__,
                    "violations": int(result_series.sum()),
                    "violations_mask": result_series,
                    "code": fixed_code,
                    "recovered_from": result["error"],
                }

                logger.info("Fixed note succeeded: %s (%d violations)", func.__name__, int(result_series.sum()))

            except Exception as e:
                logger.warning("Retry failed again for note: %s — %s", note, str(e))
                repaired[note] = {
                    "id": result["id"],
                    "code": result["code"],
                    "error": str(e),
                    "recovered_from": result["error"],
                }

        return repaired, usage_dict

import json
import os
import pandas as pd
from data_quality_tool.config.logging_config import get_logger
from data_quality_tool.llm.llm_client import LLMClient

from .note_rci_agent import NoteRCIAgent

logger = get_logger()


class NoteEngine:
    def __init__(self, model: str = None, temperature: float = 1.0):
        self.llm = LLMClient(model=model, temperature=temperature)
        self.rci = NoteRCIAgent(self.llm)
        logger.info("NoteEngine initialized with model='%s' and temperature=%.2f", model or "default", temperature)

    def strip_code_fencing(self, code: str) -> str:
        lines = code.strip().splitlines()
        if lines and lines[0].strip().startswith("```"):
            return "\n".join(line for line in lines if not line.strip().startswith("```")).strip()
        return code.strip()

    def build_prompt(self, note: str, df: pd.DataFrame) -> list[dict]:
        column_info = "\n".join(f"- `{col}` ({dtype})" for col, dtype in df.dtypes.items())
        sample = df.head(10).to_markdown(index=False)

        system_message = {
            "role": "system",
            "content": (
                "You are a data quality assistant. "
                "Your task is to translate clear domain notes into Python functions that check for violations in a dataset. "
                "Only use information supported by the note, schema, or sample — avoid guessing or assumptions."
            )
        }

        user_message = {
            "role": "user",
            "content": f"""
    Write a Python function based on the domain note below.

    ---

    Domain Note:
    "{note}"

    Available Columns:
    {column_info}

    ---

    Requirements:

    - The function must:
      - Accept a pandas DataFrame `df` as input
      - Return a pandas Series of booleans where `True` means the row **violates** the rule
      - Be vectorized using standard pandas syntax (no `.apply`)
      - Use only the listed columns (with exact names)
      - Be named descriptively using snake_case (e.g., `check_value_in_range`)

    - Do not:
      - Import pandas or any library
      - Include comments, markdown, or explanations

    Only return the function code.
    """.strip()
        }

        return [system_message, user_message]

    def generate_check(self, note: str, df: list[str], use_rci: bool = False) -> tuple[str, dict]:
        logger.info("Generating check for note: %s", note)
        messages = self.build_prompt(note, df)
        raw_code, usage = self.llm.call(messages=messages)
        clean_code = self.strip_code_fencing(raw_code)

        usage_stats = usage.copy()

        if use_rci:
            logger.debug("Running RCI refinement for note.")
            rci_result = self.rci.run_note_rci(note, clean_code, df)
            clean_code = self.strip_code_fencing(rci_result["improved"])

            usage_stats = {
                "initial": usage,
                "critique": rci_result.get("critique_usage", {}),
                "improve": rci_result.get("improvement_usage", {}),
            }
        else:
            usage_stats = {"initial": usage}

        return clean_code, usage_stats

    def run(
            self,
            notes: list[str],
            df: list[str],
            cache_path: str = "note_functions.json",
            force_refresh: bool = False,
            use_rci: bool = False
    ) -> tuple[dict, dict]:  # <-- always return a tuple
        # ── load from cache ───────────────────────────────────────────
        if not force_refresh and os.path.exists(cache_path):
            logger.info("Using cached note functions from %s", cache_path)
            with open(cache_path) as f:
                checks = json.load(f)
            return checks, {}  # <-- empty usage dict to keep the tuple shape

        # ── normal generation path ───────────────────────────────────
        checks: dict = {}
        usage_stats: dict = {}

        for note in notes:
            logger.info("Processing note: %s", note)
            try:
                code, usage = self.generate_check(note, df, use_rci=use_rci)
                logger.debug("Generated function:\n%s", code)

                checks[note] = code
                usage_stats[note] = usage

            except Exception as e:
                logger.exception("Failed to generate function for note: %s — %s", note, str(e))

        with open(cache_path, "w") as f:
            json.dump(checks, f, indent=2)
            logger.info("Saved generated checks to %s", cache_path)

        return checks, usage_stats


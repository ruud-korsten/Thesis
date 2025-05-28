import os
import json
from typing import List, Dict
from .llm_client import LLMClient
from .note_rci_agent import NoteRCIAgent
from data_quality_tool.logging_config import get_logger

logger = get_logger()


class NoteEngine:
    def __init__(self, model: str = None, temperature: float = 0.2):
        self.llm = LLMClient(model=model, temperature=temperature)
        self.rci = NoteRCIAgent(self.llm)
        logger.info("NoteEngine initialized with model='%s' and temperature=%.2f", model or "default", temperature)

    def strip_code_fencing(self, code: str) -> str:
        lines = code.strip().splitlines()
        if lines and lines[0].strip().startswith("```"):
            return "\n".join(line for line in lines if not line.strip().startswith("```")).strip()
        return code.strip()

    def build_prompt(self, note: str, df_columns: List[str]) -> str:
        return f"""
You are a data quality assistant.

Based on the following domain note, generate a Python-based data quality check function that can be applied to a pandas DataFrame.

---

Note:
"{note}"

Available Columns:
{', '.join(df_columns)}

The output must be a **Python function** that:

- Takes a pandas DataFrame as input (`df`)
- Returns a pandas Series of booleans where `True` means a row violates the check
- Uses only standard pandas and Python logic (no external libraries)

Don't import pandas. Just return the function.
Name the function descriptively. Do not include markdown or explanations.
""".strip()

    def generate_check(self, note: str, df_columns: List[str], use_rci: bool = False) -> str:
        logger.info("Generating check for note: %s", note)
        prompt = self.build_prompt(note, df_columns)
        raw_code = self.llm.call(prompt)
        clean_code = self.strip_code_fencing(raw_code)

        if use_rci:
            logger.debug("Running RCI refinement for note.")
            result = self.rci.run_note_rci(note, clean_code, df_columns)
            return self.strip_code_fencing(result["improved"])
        return clean_code

    def run(self, notes: List[str], df_columns: List[str], cache_path: str = "note_functions.json",
            force_refresh: bool = False, use_rci: bool = False) -> Dict[str, str]:
        if not force_refresh and os.path.exists(cache_path):
            logger.info("Using cached note functions from %s", cache_path)
            with open(cache_path, "r") as f:
                return json.load(f)

        checks = {}
        for note in notes:
            logger.info("Processing note: %s", note)
            try:
                code = self.generate_check(note, df_columns, use_rci=use_rci)
                logger.debug("Generated function:\n%s", code)
                checks[note] = code
            except Exception as e:
                logger.exception("Failed to generate function for note: %s — %s", note, str(e))

        with open(cache_path, "w") as f:
            json.dump(checks, f, indent=2)
            logger.info("Saved generated checks to %s", cache_path)

        return checks

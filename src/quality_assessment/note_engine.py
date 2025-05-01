import os
from typing import List, Dict
from .llm_client import LLMClient
import json

class NoteEngine:
    def __init__(self, model: str = None, temperature: float = 0.3):
        self.llm = LLMClient(model=model, temperature=temperature)

    def strip_code_fencing(self, code: str) -> str:
        """Remove markdown code block formatting from LLM output."""
        lines = code.strip().splitlines()
        if lines and lines[0].strip().startswith("```"):
            return "\n".join(line for line in lines if not line.strip().startswith("```")).strip()
        return code.strip()

    def build_prompt(self, note: str, df_columns: List[str]) -> str:
        return f"""
You are a data quality assistant.

Based on the following domain note, generate a Python-based data quality check function that can be applied to a pandas DataFrame.

---

### Note:
\"{note}\"

### Available Columns:
{', '.join(df_columns)}

The output must be a **Python function** that:

- Takes a pandas DataFrame as input (`df`)
- Returns a pandas Series of booleans where `True` means a row violates the check
- Uses only standard pandas and Python logic (no external libraries)

Don't import pandas as pd, just return the function.
Name the function descriptively. Do not add markdown or explanations — return only the Python code.
""".strip()

    def generate_check(self, note: str, df_columns: List[str]) -> str:
        prompt = self.build_prompt(note, df_columns)
        raw_code = self.llm.call(prompt)
        return self.strip_code_fencing(raw_code)

    def run(self, notes: List[str], df_columns: List[str], cache_path: str = "note_functions.json",
            force_refresh: bool = False) -> Dict[str, str]:
        if not force_refresh and os.path.exists(cache_path):
            print(f"Using cached note functions from {cache_path}")
            with open(cache_path, "r") as f:
                return json.load(f)

        checks = {}
        for note in notes:
            print(f"Generating check for note: {note}")
            code = self.generate_check(note, df_columns)
            checks[note] = code

        with open(cache_path, "w") as f:
            json.dump(checks, f, indent=2)

        return checks

import os
import json
import pandas as pd
from data_quality_tool.llm.llm_client import LLMClient
from data_quality_tool.config.logging_config import get_logger

logger = get_logger()

class ExplainViolation:
    def __init__(self, dataset_name: str, run_timestamp: str, base_dir: str = "artifacts/runs"):
        self.run_dir = os.path.join(base_dir, dataset_name, run_timestamp)
        logger.info("Initializing ExplainViolation from run directory: %s", self.run_dir)

        self.df = self._load_csv("dataset.csv")
        self.mask = self._load_csv("prediction_mask.csv")
        self.rules = self._load_json("rules.json")
        self.note_functions = self._load_json("note_functions.json")
        self.notes = self._generate_note_index_mapping(self.note_functions)
        self.llm = LLMClient()

        logger.info("Loaded dataset with shape %s and prediction mask with shape %s", self.df.shape, self.mask.shape)

    def _generate_note_index_mapping(self, notes_dict):
        """Assign note IDs and preserve mapping to descriptions."""
        notes = []
        for idx, (desc, _) in enumerate(notes_dict.items()):
            notes.append({
                "id": f"N{idx+1:03}",
                "description": desc
            })
        return notes

    def _load_csv(self, filename):
        path = os.path.join(self.run_dir, filename)
        logger.debug("Loading CSV file: %s", path)
        try:
            return pd.read_csv(path)
        except Exception as e:
            logger.error("Failed to load CSV file %s: %s", path, e)
            raise

    def _load_json(self, filename):
        path = os.path.join(self.run_dir, filename)
        logger.debug("Loading JSON file: %s", path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Could not load JSON file %s: %s", path, e)
            return {}

    def explain_violation(self, index: int) -> str:
        if index >= len(self.mask):
            return f"Index {index} is out of bounds."

        row_mask = self.mask.loc[index]
        row_data = self.df.loc[index]

        violations = {}
        for col in row_mask.index:
            if isinstance(row_mask[col], str) and row_mask[col].strip():
                violations[col] = [v.strip() for v in row_mask[col].split(",")]

        if not violations:
            return f"Row {index} has no violations."

        details = []
        for col, ids in violations.items():
            for vid in ids:
                if vid.startswith("R"):
                    desc = self._find_rule_desc(vid)
                elif vid.startswith("N"):
                    desc = self._find_note_desc(vid)
                else:
                    desc = "Unknown violation"
                details.append(f"- Column `{col}` violated {vid}: {desc} (Value: {row_data[col]})")

        prompt = f"""
A data row has been flagged for quality issues. Here is the context:

### Row Index: {index}
{row_data.to_frame().to_markdown()}

### Detected Violations:
{chr(10).join(details)}

Generate a simple, user-friendly explanation summarizing **why** this row was flagged and what might be wrong.
Look at the type of data and try to give a clear indication of what the cause of the issue might be.
        """

        logger.debug("LLM prompt:\n%s", prompt)
        explanation = self.llm.call(prompt)
        logger.info("Generated explanation: %s", explanation.strip())
        return explanation.strip()

    def _find_rule_desc(self, rule_id: str) -> str:
        return next((r.get("message", "No message") for r in self.rules if r.get("id") == rule_id), "Rule not found")

    def _find_note_desc(self, note_id: str) -> str:
        return next((n for n in self.notes if n.get("id") == note_id), "Note description not found")

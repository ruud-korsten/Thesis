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
        self.dataset_name = dataset_name
        self.base_dir = base_dir

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
Don't answer with technically advanced language, make sure that domain experts without any technical knowledge about data can very easily understand your explanation.
"""

        logger.debug("LLM prompt:\n%s", prompt)
        explanation = self.llm.call(prompt)
        logger.info("Generated explanation: %s", explanation[0])
        return explanation[0]

    def _find_rule_desc(self, rule_id: str) -> str:
        return next((r.get("message", "No message") for r in self.rules if r.get("id") == rule_id), "Rule not found")

    def _find_note_desc(self, note_id: str) -> str:
        return next((n for n in self.notes if n.get("id") == note_id), "Note description not found")

    def explain_violation_with_history(self, index: int, num_prev_runs: int = 5) -> str:
        domain_path = os.path.join(self.base_dir, self.dataset_name)
        run_paths = sorted(os.listdir(domain_path))[-num_prev_runs:]

        row_history = []

        logger.debug("Index: %s", index)
        logger.debug("Domain path: %s", domain_path)
        logger.debug("Run paths: %s", run_paths)

        for run_id in run_paths:
            run_path = os.path.join(domain_path, run_id)
            try:
                df = pd.read_csv(os.path.join(run_path, "dataset.csv"), index_col=0)
                mask = pd.read_csv(os.path.join(run_path, "prediction_mask.csv"), index_col=0)
                rule_summary = pd.read_csv(os.path.join(run_path, "rule_summary.csv"))
                note_summary = pd.read_csv(os.path.join(run_path, "note_summary.csv"))
            except Exception as e:
                logger.warning("Skipping run %s due to read error: %s", run_id, e)
                continue

            if index not in df.index:
                logger.debug("Index %s not found in run %s", index, run_id)
                continue

            if index >= len(df) or index >= len(mask):
                logger.warning("Index %s is out of bounds for either df or mask", index)
                return f"Index {index} is out of bounds."

            row_data = df.iloc[index]
            mask_data = mask.iloc[index]

            violations = {}

            logger.debug("Run ID: %s", run_id)
            logger.debug("Row data: %s", row_data.to_dict())
            logger.debug("Mask data: %s", mask_data.to_dict())

            for col_name, flag in mask_data.items():
                if not isinstance(flag, str) or not flag.strip():
                    continue

                flagged_by = []
                for vid in [v.strip() for v in flag.split(",")]:
                    if vid.startswith("R"):
                        row = rule_summary[rule_summary["Rule ID"] == vid]
                        desc = row["Description"].values[0] if not row.empty else "No description"
                        flagged_by.append(f"{vid} (Rule: {desc})")
                    elif vid.startswith("N"):
                        # We don’t have note_id — use position instead
                        note_idx = int(vid[1:]) - 1  # N001 -> 0
                        if 0 <= note_idx < len(note_summary):
                            desc = note_summary.iloc[note_idx]["Note"]
                        else:
                            desc = "No description"
                        flagged_by.append(f"{vid} (Note: {desc})")
                    else:
                        flagged_by.append(f"{vid} (Unknown type)")

                violations[col_name] = {
                    "value": row_data.get(col_name, ""),
                    "flagged_by": ", ".join(flagged_by)
                }

            logger.debug("Violations in run %s: %s", run_id, violations)
            row_history.append({
                "run_id": run_id,
                "violations": violations,
                "row_data": row_data.to_dict()
            })

        if not row_history:
            return f"No history found for row {index}."

        prompt = """You are a data quality expert. Your task is to explain why a specific row in a dataset has been flagged with data quality violations in the most recent run.

        Below is the history of the row over time, across multiple validation runs. Each run includes the row's values and any flagged violations.\n\n"""

        for run in row_history:
            prompt += f"Run ID: {run['run_id']}\n"
            if run['violations']:
                prompt += "Violations:\n"
                for col, details in run['violations'].items():
                    prompt += f"- Column: {col} | Value: {details['value']} | Flagged by: {details['flagged_by']}\n"
            else:
                prompt += "Violations: None\n"

            prompt += "Row Values:\n"
            row = run.get("row_data", {})
            for col, val in row.items():
                prompt += f"{col}: {val}\n"
            prompt += "\n---\n\n"

        prompt += """Analyze whether the values or the checks changed over time. Focus on what changed in the row, and how the applied rules or notes contributed to the most recent flag. Then explain the most likely root cause.
        """

        prompt += """
        Your response should be concise and focused.

        - Start with the most likely root cause in a single sentence.
        - Then briefly explain what changed and how it triggered the rule or note.
        - Do not repeat the row values unless necessary.
        - Be clear and avoid generic advice.
        - The aim of the response is to guide the user on how to solve the problem.
        - Also think of the root cause.
        - If the a value changed overtime, mention at which time (using run_ID) this happened.
        - Don't answer with technically advanced language, make sure that domain experts without technical any knowledge about data can easily understand your explanation.
        
        Example format:
        Root cause: [Short summary]
        Explanation: [Key changes and logic that caused the violation]
        """

        logger.info("Sending row violation history to LLM. Prompt:\n%s", prompt)
        explanation = self.llm.call(prompt)
        logger.info("Generated explanation: %s", explanation[0])
        return explanation[0]

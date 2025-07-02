import os
import json
import pandas as pd
from data_quality_tool.llm.llm_client import LLMClient
from data_quality_tool.config.logging_config import get_logger

logger = get_logger()

class FinalValidator:
    def __init__(self, run_dir: str, dataset: pd.DataFrame = None):
        self.run_dir = run_dir
        self.llm = LLMClient()
        self.dataset = dataset
        self.rule_summary = self._load_csv("rule_summary.csv")
        self.note_summary = self._load_csv("note_summary.csv")

    def _load_csv(self, filename: str) -> pd.DataFrame:
        path = os.path.join(self.run_dir, filename)
        try:
            df = pd.read_csv(path)
            logger.info("Loaded %s with shape %s", filename, df.shape)
            return df
        except Exception as e:
            logger.error("Failed to load CSV %s: %s", path, e)
            return pd.DataFrame()

    def validate(self) -> dict:
        if self.dataset is None or self.dataset.empty:
            logger.warning("Dataset is missing or empty — skipping validation.")
            return {"response": "Dataset not available for validation.", "usage": {}}

        row_count = len(self.dataset)
        summary_stats = self.dataset.describe(include='all').transpose().to_markdown()

        rules_summary = (
            self.rule_summary.to_markdown(index=False)
            if not self.rule_summary.empty
            else "No rule summary available."
        )
        notes_summary = (
            self.note_summary.to_markdown(index=False)
            if not self.note_summary.empty
            else "No note summary available."
        )

        system_message = (
            "You are a senior data quality analyst. Your job is to verify whether the given data quality rules and notes "
            "make sense given the structure and statistics of the dataset. You are provided with the dataset shape, "
            "summary statistics, and summaries of rule and note violations. Critically assess if the number of violations compared to the size "
            "of the dataset makes sense (flagging all rows as violations doesn't make sense). "
            "Don't answer with technically advanced language, make sure that domain experts without any technical knowledge about data can very easily understand it."
            "Never use emojis in your answer."
        )

        user_message = f"""
    ### Dataset Summary
    - Total rows: {row_count}

    ### Summary Statistics
    {summary_stats}

    ### Rule Violations Summary
    {rules_summary}

    ### Note Violations Summary
    {notes_summary}

    Evaluate the validity of the rules and notes. Identify any rules or notes that seem incorrect, misleading, redundant, or poorly defined. 
    Be specific about which entries you are referring to. Conclude with clear recommendations or confirmations.
    If a Rule causes an error, the rule is refactored into a note. This way it can still be evaluated. If you come across any rules with an erroneous message, check if there is a note that matches its description. Mention this in the final assessment.
    """.strip()

        logger.info("Sending validation prompt to LLM...")
        try:
            result, usage = self.llm.call(messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ])
            logger.info("Validation result received. Tokens used: %s | Cost: $%.5f",
                        usage.get("total_tokens", 0), usage.get("estimated_cost", 0.0))

            return result.strip(), usage

        except Exception as e:
            logger.error("LLM validation failed: %s", e)
            return {
                "response": "Validation failed due to an LLM error.",
                "usage": {}
            }

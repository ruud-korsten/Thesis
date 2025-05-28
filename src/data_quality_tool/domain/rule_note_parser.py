import json
import os
import pandas as pd
import re

from data_quality_tool.llm.llm_client import LLMClient
from data_quality_tool.llm.prompt_builder import build_rule_extraction_prompt
from .rule_rci_agent import RuleRCIAgent
from data_quality_tool.config.logging_config import get_logger

logger = get_logger()


class RuleParser:
    def __init__(self, model: str = None, temperature: float = 0.2):
        self.llm = LLMClient(model=model, temperature=temperature)
        self.rci = RuleRCIAgent(self.llm)
        logger.info("RuleParser initialized with model='%s' and temperature=%.2f", model or "default", temperature)

    def split_llm_response(self, response: str) -> tuple[list[dict], list[str]]:
        try:
            if not response or "### Structured Rules" not in response or "### Other Insights" not in response:
                raise ValueError("Response missing expected headers.")

            rules_section = response.split("### Other Insights")[0].split("### Structured Rules")[-1].strip()

            # Remove code fencing
            if rules_section.startswith("```") or rules_section.startswith("'''"):
                lines = rules_section.splitlines()
                if len(lines) >= 3:
                    rules_section = "\n".join(lines[1:-1]).strip()

            rules_section = re.sub(r"//.*", "", rules_section)
            insights_section = response.split("### Other Insights")[-1].strip()

            rules = json.loads(rules_section)
            insights = [line.strip("- ").strip() for line in insights_section.splitlines() if line.strip()]

            logger.info("Successfully parsed %d rules and %d insights", len(rules), len(insights))
            return rules, insights

        except Exception as e:
            logger.error("Failed to parse LLM output: %s", str(e))
            return [], [f"Failed to parse: {str(e)}"]

    def parse_rules(
            self,
            rules_path: str,
            df: pd.DataFrame,
            use_rci: bool = True,
            cache_path: str = "dq_rules.json",
            force_refresh: bool = False
    ) -> tuple[list[dict], list[str]]:
        if not force_refresh and os.path.exists(cache_path):
            logger.info("Using cached rules from %s", cache_path)
            with open(cache_path) as f:
                rules = json.load(f)
            if os.path.exists("dq_notes.json"):
                with open("dq_notes.json") as f:
                    notes = json.load(f)
            else:
                notes = []
            return rules, notes

        if df.empty:
            raise ValueError("Provided DataFrame is empty.")

        with open(rules_path, encoding="utf-8") as f:
            domain_text = f.read().strip()

        prompt = build_rule_extraction_prompt(domain_text, df.columns.tolist())
        logger.info("Running rule extraction...")

        if use_rci:
            result = self.rci.run_rci_pipeline(prompt, domain_text, df.columns.tolist())
            raw_response = result["improved_output"]
        else:
            raw_response = self.llm.call(prompt)

        rules, notes = self.split_llm_response(raw_response)

        with open(cache_path, "w") as f:
            json.dump(rules, f, indent=2)
        with open("dq_notes.json", "w") as f:
            json.dump(notes, f, indent=2)

        logger.info("Extracted %d rules and %d notes", len(rules), len(notes))
        return rules, notes

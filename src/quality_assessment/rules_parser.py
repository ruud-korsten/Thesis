import json
from typing import List, Dict
import pandas as pd
from .prompt_builder import build_rule_extraction_prompt
from .rci_agent import RCIAgent
from .llm_client import LLMClient
import os

class RuleParser:
    def __init__(self, model: str = None, temperature: float = 0.2):
        self.llm = LLMClient(model=model, temperature=temperature)
        self.rci = RCIAgent(self.llm)

    def split_llm_response(self, response: str) -> tuple[list[Dict], list[str]]:
        import re

        try:
            if not response or "### Structured Rules" not in response or "### Other Insights" not in response:
                raise ValueError("Response missing expected headers.")

            rules_section = response.split("### Other Insights")[0].split("### Structured Rules")[-1].strip()

            # Remove triple quotes or backticks
            if rules_section.startswith("```") or rules_section.startswith("'''"):
                lines = rules_section.splitlines()
                if len(lines) >= 3:
                    rules_section = "\n".join(lines[1:-1]).strip()

            # Remove inline JS-style comments
            rules_section = re.sub(r"//.*", "", rules_section)

            insights_section = response.split("### Other Insights")[-1].strip()
            rules = json.loads(rules_section)
            insights = [line.strip("- ").strip() for line in insights_section.splitlines() if line.strip()]

            return rules, insights

        except Exception as e:
            print(f"Failed to parse LLM output: {e}")
            return [], [f"Failed to parse: {str(e)}"]

    from typing import List, Dict, Tuple

    def parse_rules(
            self,
            rules_path: str,
            df: pd.DataFrame,
            use_rci: bool = True,
            cache_path: str = "dq_rules.json",
            force_refresh: bool = False
    ) -> Tuple[List[Dict], List[str]]:
        if not force_refresh and os.path.exists(cache_path):
            print(f"Using cached rules from {cache_path}")
            with open(cache_path, "r") as f:
                rules = json.load(f)
            if os.path.exists("dq_notes.json"):
                with open("dq_notes.json", "r") as f:
                    notes = json.load(f)
            else:
                notes = []
            return rules, notes

        if not df.empty:
            with open(rules_path, "r", encoding="utf-8") as f:
                domain_text = f.read().strip()

            prompt = build_rule_extraction_prompt(domain_text, df.columns.tolist())
            print("Running rule extraction...")

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

            print(f"Extracted {len(rules)} rules and {len(notes)} notes.")
            return rules, notes
        else:
            raise ValueError("Provided DataFrame is empty.")

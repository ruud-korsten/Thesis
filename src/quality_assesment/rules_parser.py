import openai
import os
import json
from typing import List, Dict
from dotenv import load_dotenv
import pandas as pd

class RuleParser:
    def __init__(self, model: str = None, temperature: float = 0.2):
        load_dotenv()

        openai.api_key = os.getenv("OPENAI_API_KEY")

        # Allow override or fallback to environment value
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature

    def load_rules_txt(self, path: str) -> List[str]:
        """Step 1: Load natural language rules from a text file."""
        with open(path, "r") as file:
            return [line.strip() for line in file if line.strip()]

    def generate_prompt(self, rule_text: str, rule_id: str, df: pd.DataFrame = None) -> str:
        column_list_str = ", ".join(df.columns.tolist()) if df is not None else "Not provided"
        return f"""
    You are a data quality assistant. Convert the following rule into a structured JSON object.

    Use only the following structure where applicable:

    - id: string (e.g., "R001")
    - name: short name for the rule
    - type: one of ["range", "not_null", "frequency", "conditional", "pattern"]
    - column: the name of the column the rule applies to (or 'columns' for multiple)
    - condition:
        - For range rules: use an object like {{ "min": 0, "max": 100 }} (omit if not needed)
        - For pattern rules: use a string representing the regex pattern
        - For conditional rules: use:
            "condition": {{
                "if": {{ "column": ..., "equals": ... }},
                "then": {{ "column": ..., "condition": {{ "min": ..., "max": ... }} }}
            }}
    - expected_interval: (for frequency rules only, e.g., "15 minutes")
    - group_by: (optional, e.g., "sensor_id")
    - message: explanation of the rule

    Ensure that range conditions are always structured using "min" and/or "max" — do not use expressions like ">= 0".

    Rule ID: {rule_id}
    Rule: "{rule_text}"
    
    The dataset contains the following columns:
    [{column_list_str}]
    
    Make sure that any column used in the rules is an actual column in the dataset.

    Return only the JSON object — no extra formatting, no comments, and no markdown.
        """.strip()

    def call_llm(self, prompt: str) -> Dict:
        """Step 3: Execute prompt with OpenAI API and return parsed JSON."""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            content = response['choices'][0]['message']['content']
            return json.loads(content)
        except Exception as e:
            return {"error": str(e), "raw_output": content}

    def parse_rules(
            self,
            rules_path: str,
            cache_path: str = "dq_rules.json",
            force_refresh: bool = False,
            df: pd.DataFrame = None,
    ) -> List[Dict]:
        """
        Parse rules from a text file into structured JSON.
        If a cached version exists and force_refresh is False, reuse the cached file.

        Args:
            rules_path: Path to the rules.txt file
            cache_path: Path to store or load the JSON rules
            force_refresh: If True, re-parse rules even if cache exists

        Returns:
            List of structured rule dictionaries
        """
        if not force_refresh and os.path.exists(cache_path):
            print(f"Using cached rules from {cache_path}")
            with open(cache_path, "r") as f:
                return json.load(f)

        print(f"Parsing rules from {rules_path} using model: {self.model}")
        raw_rules = self.load_rules_txt(rules_path)
        parsed_rules = []

        for i, rule_text in enumerate(raw_rules, start=1):
            rule_id = f"R{str(i).zfill(3)}"
            print(f"Parsing rule {rule_id}: {rule_text}")
            prompt = self.generate_prompt(rule_text, rule_id, df)
            result = self.call_llm(prompt)
            parsed_rules.append(result)

        self.save_rules_to_json(parsed_rules, cache_path)
        return parsed_rules

    def save_rules_to_json(self, rules: List[Dict], output_path: str):
        """Optional: Save parsed rules to a JSON file."""
        with open(output_path, "w") as f:
            json.dump(rules, f, indent=2)
        print(f"Saved {len(rules)} rules to {output_path}")

class RCIAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    def build_critique_prompt(self, output: str, domain_text: str, df_columns: list[str]) -> str:
        return f"""
You are a senior data quality engineer. Your task is to critique a rule extraction output based on the provided domain knowledge, dataset schema, and the known capabilities of the rule engine.

Your responsibilities:

1. Identify missing rules that are clearly implied in the domain knowledge but were not extracted.
2. Flag invalid rules: unsupported types, malformed structure, or use of unknown columns.
3. Correct misclassifications: e.g., enforceable rules placed in "Other Insights".
4. Ensure schema alignment: rules must only use columns from the dataset schema.
5. Suggest improvements: naming, formatting, or structure where applicable.

---

### Rule Engine Capabilities

Only the following rule types are supported:

- "range": Numeric column. Format:
  "condition": {{ "min": ..., "max": ... }}

- "not_null": One column. Checks for missing values. Format:
  "condition": {{}}

- "pattern": One string column with regex format. Format:
  "condition": {{ "regex": "..." }}

- "conditional": Requires one IF condition and one THEN check.

  Supported formats:

  Format A:
  "condition": {{
    "if": {{ "column": "X", "equals": "value" }},
    "then": {{ "column": "Y", "not_null": true }}
  }}

  Format B:
  "condition": {{
    "if": {{ "X": "value" }},
    "then": {{ "Y": "filled" }}
  }}

  Format C (range-based THEN logic):
  "condition": {{
    "if": {{ "column": "X", "equals": "value" }},
    "then": {{
      "column": "Y",
      "condition": {{ "min": ..., "max": ... }}
    }}
  }}

Do not use shortcuts like: `"then": {{ "Y": {{ "min": 0.01 }} }}` — they are not valid.

Unsupported types: temporal, uniqueness, validity, consistency.

---

### Dataset Columns

[{", ".join(df_columns)}]

Only use the columns listed above. Rules that use other columns are invalid and should be flagged.

---

### Domain Knowledge

\"\"\"
{domain_text}
\"\"\"

---

### Rule Extraction Output to Critique

\"\"\"
{output}
\"\"\"

---

### Your Critique

Use the following structure:

#### Missing Rules
...

#### Invalid Rules
...

#### Misclassified Insights
...

#### Suggestions
...
""".strip()

    def build_improvement_prompt(self, original: str, critique: str, domain_text: str, df_columns: list[str]) -> str:
        return f"""
You are a data validation assistant.

Your task is to improve a rule extraction output based on a structured critique.
You must:
- Add any missing rules described in the critique
- Fix or remove any invalid or hallucinated rules
- Reclassify insights that should be structured rules
- Improve rule names, formatting, or messages for clarity and consistency

---

### Dataset Columns

[{", ".join(df_columns)}]

Use only the columns listed above. Do not invent or reference any column not present in this list.

---

### Rule Format

Each structured rule must follow this JSON schema:

- "id": e.g., "R001"
- "name": a short human-readable title
- "type": one of the four supported types
- "column" or "columns": the column(s) the rule applies to
- "condition": a rule-specific logic object (e.g., {{ "min": 0 }})
- "group_by": optional
- "message": a short explanation of what the rule means

---

### Allowed Rule Types

- "range": For numeric values.
  Format: "condition": {{ "min": ..., "max": ... }}

- "not_null": One column. Flags missing values.
  Format: "condition": {{}}

- "pattern": For regex/fixed-format.
  Format: "condition": {{ "regex": "..." }}

- "conditional": Use one IF and one THEN condition.

  Use one of the following formats:

  Format A:
  "condition": {{
    "if": {{ "column": "X", "equals": "value" }},
    "then": {{ "column": "Y", "not_null": true }}
  }}

  Format B:
  "condition": {{
    "if": {{ "X": "value" }},
    "then": {{ "Y": "filled" }}
  }}

  Format C (for range logic):
  "condition": {{
    "if": {{ "column": "X", "equals": "value" }},
    "then": {{
      "column": "Y",
      "condition": {{ "min": ..., "max": ... }}
    }}
  }}

Do not use flattened THEN structures like: "then": {{ "Y": {{ "min": 0 }} }} — these are not valid and will be rejected.

---

### Domain Knowledge

\"\"\"
{domain_text}
\"\"\"

---

### Critique

\"\"\"
{critique}
\"\"\"

---

### Original Rule Extraction

\"\"\"
{original}
\"\"\"

---

### Your Response

Return your improved output using the format below:

### Structured Rules
[ ... valid JSON array ... ]

### Other Insights
- Insight 1
- Insight 2
...
""".strip()


    def run_rci_pipeline(self, prompt: str, domain_text: str, df_columns: list[str]) -> dict:
        initial_output = self.llm.call(prompt)
        # print("Initial output:")
        # print(initial_output)
        critique_prompt = self.build_critique_prompt(initial_output, domain_text, df_columns)
        critique = self.llm.call(critique_prompt)
        # print("Critique:")
        # print(critique)
        improve_prompt = self.build_improvement_prompt(initial_output, critique, domain_text, df_columns)
        improved_output = self.llm.call(improve_prompt)
        # print("Improved:")
        # print(improved_output)

        return {
            "initial_output": initial_output,
            "critique": critique,
            "improved_output": improved_output
        }

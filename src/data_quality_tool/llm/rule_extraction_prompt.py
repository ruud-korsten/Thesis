from data_quality_tool.config.logging_config import get_logger

logger = get_logger()

def build_rule_extraction_messages(domain_text: str, df_columns: list[str]) -> list[dict]:
    """
    Constructs the messages payload for OpenAI's ChatCompletion API with system and user roles.

    Returns:
        List of messages with roles 'system' and 'user'.
    """
    column_list_str = ", ".join(df_columns)

    logger.debug(
        "Building rule extraction messages with %d columns and domain text length %d",
        len(df_columns),
        len(domain_text),
    )

    system_message = {
        "role": "system",
        "content": (
            "You are a data quality assistant. You extract structured validation rules and "
            "surface natural-language insights from domain-specific documentation. Always follow "
            "the exact output format provided. Do not infer or invent data. Be precise, JSON-compliant, "
            "and only use column names explicitly listed in the prompt."
        )
    }

    user_message = {
        "role": "user",
        "content": f"""
Your task is to extract structured validation rules and separate natural-language insights from the domain knowledge provided below.

---

### Rule Types

Extract all **clear, enforceable validation rules**. Each rule must match one of the following types:

- `range`: min/max bounds on a numeric column
- `not_null`: the column must not contain missing values
- `conditional`: logic such as "if column A = value, then column B must..."
- `pattern`: regex or fixed-format constraints (e.g., 18-digit ID)

If a rule is implied (e.g., "must not be negative"), still extract it in structured form.

If a rule does **not** clearly follow one of the types listed above, **do not extract it** as a structured rule.

---

### Rule Format

Each structured rule must follow this JSON schema:

- `id`: e.g., "R001"
- `name`: a short human-readable title
- `type`: one of the four supported types
- `column` or `columns`: the column(s) the rule applies to
- `condition`: a rule-specific logic object (e.g., {{ "min": 0 }})
- `message`: a short explanation of the rule in plain language
- `provenance`: an explanation on what the rule is based on

---

### Other Insights

List any helpful expectations or context that do **not** fit the structured rule format. These might include:

- Sensor behavior or irregular patterns
- Data reliability concerns
- Edge cases or operational notes
- Extreme outliers

Also include **any rule that does not follow the allowed types** in this section, written as a plain-language sentence.

---

### Dataset Columns

[{column_list_str}]

Use **only** the column names listed above in all rules.

Do not invent columns like "source" or "sensor_id" unless they appear in the dataset.

Any rule that references unknown columns should be skipped or rewritten using valid names.

Use the provided domain knowledge to guide the extraction process.

---

### Domain Knowledge

\"\"\"
{domain_text}
\"\"\"

---

### Your response must contain exactly the following two sections:

### Structured Rules
[ ... valid JSON array ... ]

### Other Insights
- Insight 1
- Insight 2
...
""".strip()
    }

    return [system_message, user_message]

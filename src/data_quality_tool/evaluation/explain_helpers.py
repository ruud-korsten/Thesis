import os
import json
from data_quality_tool.llm.llm_client import LLMClient

def explain_selected_rule_or_note(run_path: str, selected_type: str, selected_id: str) -> str:
    dataset_name = os.path.basename(os.path.dirname(run_path))
    run_timestamp = os.path.basename(run_path)
    run_dir = os.path.join("artifacts", "runs", dataset_name, run_timestamp)

    # Load domain knowledge
    domain_path = os.path.join(run_dir, "domain.txt")
    domain_knowledge = ""
    if os.path.exists(domain_path):
        with open(domain_path, "r", encoding="utf-8") as f:
            domain_knowledge = f.read()

    if selected_type == "Rule":
        try:
            with open(os.path.join(run_dir, "rules.json"), "r", encoding="utf-8") as f:
                rules = json.load(f)
        except Exception as e:
            return f"Failed to load rules.json: {e}"

        rule = next((r for r in rules if r.get("id") == selected_id), None)
        if not rule:
            return f"Rule {selected_id} not found."

        description = rule.get("message", "")
        provenance = rule.get("provenance", "No provenance available.")

    elif selected_type == "Note":
        try:
            with open(os.path.join(run_dir, "note_functions.json"), "r", encoding="utf-8") as f:
                note_functions = json.load(f)
        except Exception as e:
            return f"Failed to load note_functions.json: {e}"

        note_keys = list(note_functions.keys())
        index = int(selected_id[1:]) - 1  # e.g., "N001" → 0
        if 0 <= index < len(note_keys):
            description = note_keys[index]
            provenance = "Derived from general note logic using LLM."
        else:
            return f"Note {selected_id} not found."

    else:
        return "Invalid type: must be 'Rule' or 'Note'."

    # Construct the prompt
    prompt = f"""
You are a data quality expert. Your task is to explain the logic of the following {selected_type.lower()} to a business user with no technical background.

### Context
- Dataset: {dataset_name}
- {selected_type} ID: {selected_id}
- Description: {description}
- Provenance: {provenance}
- Domain Knowledge:
{domain_knowledge.strip()}

Explain clearly and simply:
- What this {selected_type.lower()} is checking.
- Why it may have been created for this dataset (based on the provided domain knowledge).
- Clearly highlight the provenance based on the domain knowledge).
- What kind of data problems it is designed to catch.
- Provide the explanation in simple, academic-level language.
- Make sure it is understandable by any user, with or without technical knowledge.

### Output Format
Write only the explanation.
"""

    # Debugging
    print("=" * 20 + " Prompt Sent to LLM " + "=" * 20)
    print(prompt)
    print("=" * 60)

    # Call LLM
    llm = LLMClient()
    try:
        response = llm.call(prompt)
        return response[0]
    except Exception as e:
        return f"Error generating explanation: {e}"

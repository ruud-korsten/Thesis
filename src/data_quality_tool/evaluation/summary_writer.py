import pandas as pd
import os
from data_quality_tool.config.logging_config import get_logger

logger = get_logger()

def save_rule_note_summary(rule_reports: dict, note_results: dict, output_dir="artifacts"):
    os.makedirs(output_dir, exist_ok=True)

    # === Rule Summary ===
    rule_rows = []
    for rule_id, report in rule_reports.items():
        rule_rows.append({
            "Rule ID": rule_id,
            "Violations": report.get("violations", 0),
            "Description": report.get("message", "")
        })
    rule_df = pd.DataFrame(rule_rows)
    rule_csv_path = os.path.join(output_dir, "rule_summary.csv")
    rule_df.to_csv(rule_csv_path, index=False)
    logger.info("Saved rule summary to %s", rule_csv_path)

    # === Note Summary ===
    note_rows = []

    # Handle passed notes
    for note_text, result in note_results.get("passed", {}).items():
        note_rows.append({
            "Note": note_text,
            "Violations": result.get("violations", 0),
            "Function": result.get("function_name", result.get("id", "N/A"))
        })

    # Handle failed notes
    for note_text, result in note_results.get("failed", {}).items():
        note_rows.append({
            "Note": note_text,
            "Violations": "ERROR",
            "Function": "N/A"
        })

    note_df = pd.DataFrame(note_rows)
    note_csv_path = os.path.join(output_dir, "note_summary.csv")
    note_df.to_csv(note_csv_path, index=False)
    logger.info("Saved note summary to %s", note_csv_path)

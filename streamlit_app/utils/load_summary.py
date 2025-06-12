import os
import pandas as pd
import json

def load_violation_summary(run_path, run_date):
    data = pd.read_csv(os.path.join(run_path, "dataset.csv"))

    shape = data.shape
    columns = data.columns.tolist()
    sample_rows = data.head(10)
    summary_stats = data.describe(include="all").transpose()

    missing = data.isna().sum()
    percent_missing = 100 * missing / len(data)
    missing_summary = pd.DataFrame({
        "missing_values": missing,
        "percent_missing": percent_missing
    }).sort_values("missing_values", ascending=False)

    duplicate_flags = data.duplicated()
    duplicate_count = duplicate_flags.sum()
    duplicate_summary = {
        "total_rows": len(data),
        "duplicate_rows": int(duplicate_count),
        "percent": round(100 * duplicate_count / len(data), 4)
    }
    duplicate_examples = data[duplicate_flags].head(10)

    rule_path = os.path.join(run_path, "rule_summary.csv")
    note_path = os.path.join(run_path, "note_summary.csv")

    try:
        rule_violations = pd.read_csv(rule_path) if os.path.exists(rule_path) else pd.DataFrame()
    except pd.errors.EmptyDataError:
        rule_violations = pd.DataFrame(columns=["Rule ID", "Violations", "Description"])

    try:
        note_violations = pd.read_csv(note_path) if os.path.exists(note_path) else pd.DataFrame()
    except pd.errors.EmptyDataError:
        note_violations = pd.DataFrame(columns=["Function", "Violations", "Note"])


    return {
        "overview": {
            "shape": shape,
            "columns": columns,
            "sample_rows": sample_rows,
            "summary_stats": summary_stats
        },
        "missing": missing_summary,
        "duplicates": {
            "summary": duplicate_summary,
            "examples": duplicate_examples
        },
        "rules": rule_violations,
        "notes": note_violations
    }

# ➕ Additional helpers for detailed tab

def load_dataset_and_mask(run_path):
    try:
        dataset = pd.read_csv(os.path.join(run_path, "dataset.csv"))
        pred_mask = pd.read_csv(os.path.join(run_path, "prediction_mask.csv"))
        return dataset, pred_mask
    except Exception as e:
        raise RuntimeError(f"Error loading dataset or prediction mask: {e}")

def load_summary_files(run_path):
    rule_path = os.path.join(run_path, "rule_summary.csv")
    note_path = os.path.join(run_path, "note_summary.csv")

    try:
        try:
            rule_summary = pd.read_csv(rule_path)
        except pd.errors.EmptyDataError:
            rule_summary = pd.DataFrame(columns=["Rule ID", "Violations", "Description"])

        try:
            note_summary = pd.read_csv(note_path)
        except pd.errors.EmptyDataError:
            note_summary = pd.DataFrame(columns=["Function", "Violations", "Note"])

        return rule_summary, note_summary

    except Exception as e:
        raise RuntimeError(f"Error loading summary CSVs: {e}")


def load_rule_and_note_definitions(run_path):
    try:
        with open(os.path.join(run_path, "rules.json")) as f:
            rules = json.load(f)
        with open(os.path.join(run_path, "note_functions.json")) as f:
            notes = json.load(f)
        return rules, notes
    except Exception as e:
        raise RuntimeError(f"Error loading JSON definitions: {e}")

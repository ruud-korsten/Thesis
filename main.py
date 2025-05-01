from quality_assessment.dataset_selection import load_dataset
from quality_assessment.checker import DataQualityChecker
from quality_assessment.rules_parser import RuleParser
from quality_assessment.rule_engine import RuleExecutor
from quality_assessment.rule_inspector import inspect_rule_violations
from quality_assessment.note_engine import NoteEngine
from quality_assessment.note_evaluator import NoteEvaluator
from quality_assessment.note_inspector import inspect_note_violations

import pandas as pd

# --- Helper functions ---mport path if needed

def run_basic_dq(df):
    dq = DataQualityChecker(df)
    reports = dq.run_all_checks()

    print("Missing Values:")
    print(reports['missing'].head())

    print("\nDuplicate Summary:")
    print(reports['duplicates'])

def run_rule_engine(df: pd.DataFrame, rules: list[dict]) -> tuple[pd.DataFrame, RuleExecutor]:
    executor = RuleExecutor(df)
    result_df = executor.apply_rules(rules)

    print("\nRule Violation Summary:")
    for rule_id, report in executor.reports.items():
        if "error" in report.get("message", "").lower():
            print(f"{rule_id}: ERROR - {report['message']}")
        else:
            print(f"{rule_id}: {report['violations']} violations - {report['message']}")

    return result_df, executor

def run_and_evaluate_note_engine(notes: list[str], df: pd.DataFrame) -> dict:
    if not notes:
        print("\nNo domain notes found to process.")
        return {}

    print("\nRunning Note Engine on domain insights...")
    note_engine = NoteEngine(model="gpt-4o")
    note_functions = note_engine.run(notes, df.columns.tolist(), cache_path="note_functions.json", force_refresh=False)

    print("\nEvaluating generated data quality checks...")
    evaluator = NoteEvaluator(df)
    note_results = evaluator.evaluate(note_functions)

    for note, result in note_results.items():
        print("\n---")
        print(f"Note: {note}")
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"✓ Function: {result['function_name']}")
            print(f"→ Violations flagged: {result['violations']}")
            print("Code:\n" + result["code"])

    return note_results

def main():
    df, domain_path = load_dataset("energy")

    print(df.head())
    # Traditional Checks
    run_basic_dq(df)

    parser = RuleParser(model="gpt-4o-mini")
    rules, notes = parser.parse_rules(
        rules_path=domain_path,
        cache_path="dq_rules.json",
        force_refresh=False,
        df=df,
        use_rci=True
    )

    print("\nRule Extraction Complete.")
    print(f"Extracted {len(rules)} rules. See dq_rules.json and dq_notes.json for full output.")

    result_df, executor = run_rule_engine(df, rules)
    inspect_rule_violations(result_df, executor.reports)

    #Move rules with an error to notes
    print(executor.fallback_notes)
    all_notes = notes + executor.fallback_notes
    print(all_notes)

    # Run Note Engine
    note_results = run_and_evaluate_note_engine(all_notes, df)
    inspect_note_violations(df, note_results)



if __name__ == "__main__":
    main()
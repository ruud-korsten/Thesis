from quality_assesment.db import fetch_table
from quality_assesment.checker import DataQualityChecker
from quality_assesment.rules_parser import RuleParser
from quality_assesment.rule_engine import RuleExecutor

# --- Helper functions ---

def run_basic_dq(df):
    dq = DataQualityChecker(df)

    print("Missing Values:")
    print(dq.check_missing_values())

    print("\nOutliers:")
    dq.detect_outliers_iqr_grouped(group_cols=['ean'], target_cols=['usage'])

    print("\nOutlier Summary:")
    print(dq.reports['outliers_summary_usage'].head())

    print("\nDuplicate Summary:")
    dq.check_duplicate_rows()
    print(dq.reports['duplicates'])


def run_rule_engine(df):
    parser = RuleParser()
    rules = parser.parse_rules("rules.txt", cache_path="dq_rules.json", force_refresh=False, df=df)

    executor = RuleExecutor(df)
    result_df = executor.apply_rules(rules)

    print("\nRule Violation Summary:")
    for rule_id, report in executor.reports.items():
        if "violations" in report:
            print(f"{rule_id}: {report['violations']} violations - {report['message']}")
        else:
            print(f"{rule_id}: ERROR - {report['error']}")

    selected = input("\nEnter rule IDs to inspect (comma-separated, e.g. R001,R005): ").split(",")
    for rule_id in [r.strip().upper() for r in selected if r.strip()]:
        try:
            print(f"\nViolations of {rule_id}:")
            violations_df = executor.get_violations_for_rule(rule_id=rule_id, df=result_df, show="rule")
            print(violations_df.head())
        except Exception as e:
            print(f"Could not display violations for {rule_id}: {e}")

    return result_df

def main():
    df = fetch_table("public_grafana_ovvia.energy_highfrequent")

    # Traditional Checks
    run_basic_dq(df)

    # Rule-Based Checks
    result_df = run_rule_engine(df)

if __name__ == "__main__":
    main()
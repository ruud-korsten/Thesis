from .rule_engine import RuleExecutor
import pandas as pd

def inspect_rule_violations(df: pd.DataFrame, reports: dict):
    executor = RuleExecutor(df)

    while True:
        rule_id = input("\nEnter a rule ID to inspect (e.g., R001), or press Enter to exit: ").strip().upper()
        if not rule_id:
            print("Exiting inspection.")
            break

        if rule_id not in reports:
            print(f"Rule {rule_id} not found in the report.")
            continue

        try:
            rule = reports[rule_id].get("rule", {})
            flag_col = f"violation_{rule_id}"
            if flag_col not in df.columns:
                print(f"No violation column found for {rule_id}.")
                continue

            # Identify relevant columns for display
            cols = []
            if rule.get("type") == "conditional":
                cond = rule.get("condition", {})
                if_cond = cond.get("if", {})
                then_cond = cond.get("then", {})

                if isinstance(if_cond, dict):
                    if "column" in if_cond:
                        cols.append(if_cond["column"])
                    else:
                        cols.extend(list(if_cond.keys()))

                if isinstance(then_cond, dict):
                    if "column" in then_cond:
                        cols.append(then_cond["column"])
                    else:
                        cols.extend(list(then_cond.keys()))
            else:
                col = rule.get("column") or rule.get("columns")
                if isinstance(col, str):
                    cols.append(col)
                elif isinstance(col, list):
                    cols.extend(col)

            cols = list(set([c for c in cols if c in df.columns]))  # ensure columns exist

            # Build result view
            violations = df[df[flag_col] == True]
            print(f"\nViolations for {rule_id} (showing first 5 rows):")
            print(violations[cols + [flag_col]].head())

        except Exception as e:
            print(f"Could not retrieve violations for {rule_id}: {e}")

import pandas as pd
from typing import List, Dict

from requests.utils import dict_from_cookiejar


class RuleExecutor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.reports = {}

    def apply_rule(self, rule: Dict) -> pd.Series:
        column = rule.get("column")
        rule_type = rule.get("type")
        condition = rule.get("condition")

        if rule_type == "range":
            min_val = rule.get("condition", {}).get("min")
            max_val = rule.get("condition", {}).get("max")

            series = self.df[column]
            violation_mask = pd.Series(False, index=self.df.index)

            if min_val is not None:
                violation_mask |= series < min_val
            if max_val is not None:
                violation_mask |= series > max_val

            return violation_mask

        elif rule_type == "not_null":
            return self.df[column].isnull()

        elif rule_type == "pattern":
            return ~self.df[column].astype(str).str.match(condition)

        elif rule_type == "conditional":
            cond = rule.get("condition", {})
            if_cond = cond.get("if", {})
            then_cond = cond.get("then", {})
            # Condition mask
            mask_if = self.df[if_cond["column"]] == if_cond["equals"]
            if "not_null" in then_cond and then_cond["not_null"]:
                mask_then = self.df[then_cond["column"]].isnull()
            else:
                # fallback for range or other checks
                min_val = then_cond.get("condition", {}).get("min")
                max_val = then_cond.get("condition", {}).get("max")
                series = self.df[then_cond["column"]]
                mask_then = pd.Series(False, index=self.df.index)
                if min_val is not None:
                    mask_then |= series < min_val
                if max_val is not None:
                    mask_then |= series > max_val
            return mask_if & mask_then

        elif rule_type == "frequency":
            expected_interval = rule.get("expected_interval")
            group_col = rule.get("group_by")  # Optional

            df_sorted = self.df.sort_values(by=column)

            # Compute time gaps
            if group_col:
                time_diffs = df_sorted.groupby(group_col)[column].diff()
            else:
                time_diffs = df_sorted[column].diff()

            # Parse the threshold (e.g., "15 minutes")
            threshold = pd.to_timedelta(expected_interval)

            # Find gaps larger than the threshold
            violation_mask = ((time_diffs > threshold)
                              .shift(-1)
                              .fillna(False)
                              .astype(bool))

            # Align to original DataFrame index
            return violation_mask.reindex(self.df.index, fill_value=False)


        else:
            raise NotImplementedError(f"Rule type '{rule_type}' not supported.")

    def apply_rules(self, rules: List[Dict]) -> pd.DataFrame:
        self.df = self.df.copy()  # Update internal DataFrame with flags

        for rule in rules:
            rule_id = rule.get("id", "unknown")
            try:
                violation_mask = self.apply_rule(rule)
                self.df[f"violation_{rule_id}"] = violation_mask
                self.reports[rule_id] = {
                    "message": rule.get("message", "No message provided."),
                    "violations": int(violation_mask.sum()),
                    "rule": rule
                }
            except Exception as e:
                self.reports[rule_id] = {
                    "message": rule.get("message", "No message provided."),
                    "violations": int(violation_mask.sum()),
                    "rule": rule
                }

        return self.df

    def get_violations_for_rule(
            self,
            rule_id: str,
            df: pd.DataFrame,
            show: str = "all"  # Options: "all", "rule"
    ) -> pd.DataFrame:

        flag_col = f"violation_{rule_id}"

        if flag_col not in list(df.columns):
            raise ValueError(f"Violation column '{flag_col}' not found in the DataFrame.")

        df_violations = df[df[flag_col] == True].copy()

        if show == "rule":
            rule_info = self.reports.get(rule_id, {}).get("rule", {})
            rule_columns = []

            print(rule_info)

            if rule_info.get("type") == "conditional":
                condition = rule_info.get("condition", {})
                rule_columns = [
                    condition.get("if", {}).get("column"),
                    condition.get("then", {}).get("column")
                ]
            else:
                rule_columns = [rule_info.get("column")]

            rule_columns = [col for col in rule_columns if col is not None]

            if not rule_columns:
                raise ValueError(f"No column info available for rule {rule_id} to show only relevant fields.")

            return df_violations[rule_columns + [flag_col]]

        return df_violations

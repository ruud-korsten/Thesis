from collections.abc import Callable
import pandas as pd
from data_quality_tool.logging_config import get_logger

logger = get_logger()


class RuleExecutor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.reports = {}
        logger.debug("RuleExecutor initialized with DataFrame of shape %s", self.df.shape)

        self.handlers: dict[str, Callable[[dict], pd.Series]] = {
            "range": self._handle_range,
            "not_null": self._handle_not_null,
            "pattern": self._handle_pattern,
            "conditional": self._handle_conditional,
        }

    def _handle_range(self, rule: dict) -> pd.Series:
        logger.debug("Handling 'range' rule: %s", rule.get("id"))
        columns = rule.get("columns") or [rule.get("column")]
        condition = rule.get("condition", {})
        violation_mask = pd.Series(False, index=self.df.index)

        for col in columns:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not in DataFrame.")

            # Coerce column to numeric, invalid parsing becomes NaN
            series = pd.to_numeric(self.df[col], errors="coerce")

            if "min" in condition:
                min_val = condition["min"]
                violation_mask |= series < min_val

            if "max" in condition:
                max_val = condition["max"]
                violation_mask |= series > max_val

        return violation_mask

    def _handle_not_null(self, rule: dict) -> pd.Series:
        logger.debug("Handling 'not_null' rule: %s", rule.get("id"))
        column = rule.get("column")
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not in DataFrame.")
        return self.df[column].isnull()

    def _handle_pattern(self, rule: dict) -> pd.Series:
        logger.debug("Handling 'pattern' rule: %s", rule.get("id"))
        column = rule.get("column")
        pattern = rule.get("condition")
        if isinstance(pattern, dict):
            pattern = pattern.get("regex", "")
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not in DataFrame.")
        return ~self.df[column].astype(str).str.match(pattern)

    def _handle_conditional(self, rule: dict) -> pd.Series:
        logger.debug("Handling 'conditional' rule: %s", rule.get("id"))
        cond = rule.get("condition", {})
        if_cond = cond.get("if", {})
        then_cond = cond.get("then", {})

        if isinstance(if_cond, dict) and "column" in if_cond and "equals" in if_cond:
            if_col = if_cond["column"]
            if_val = if_cond["equals"]
        elif len(if_cond) == 1:
            if_col, if_val = list(if_cond.items())[0]
        else:
            raise ValueError(f"Invalid IF condition format: {if_cond}")

        if if_col not in self.df.columns:
            raise ValueError(f"IF column '{if_col}' not found in DataFrame")

        mask_if = self.df[if_col] == if_val

        if "column" in then_cond:
            then_col = then_cond["column"]
            if then_cond.get("not_null") is True:
                mask_then = self.df[then_col].isnull()
            else:
                then_check = then_cond.get("condition", {})
                if not then_check:
                    raise ValueError(f"No THEN condition specified for column '{then_col}'")
                mask_then = pd.Series(False, index=self.df.index)
                if "min" in then_check:
                    mask_then |= self.df[then_col] < then_check["min"]
                if "max" in then_check:
                    mask_then |= self.df[then_col] > then_check["max"]
        elif len(then_cond) == 1:
            then_col, cond_body = list(then_cond.items())[0]
            if then_col not in self.df.columns:
                raise ValueError(f"THEN column '{then_col}' not found in DataFrame")
            mask_then = pd.Series(False, index=self.df.index)
            if "not_null" in cond_body and cond_body["not_null"] is True:
                mask_then = self.df[then_col].isnull()
            else:
                if "min" in cond_body:
                    mask_then |= self.df[then_col] < cond_body["min"]
                if "max" in cond_body:
                    mask_then |= self.df[then_col] > cond_body["max"]
        else:
            raise ValueError(f"Unsupported THEN condition format: {then_cond}")

        return mask_if & mask_then

    def apply_rule(self, rule: dict) -> pd.Series:
        rule_type = rule.get("type")
        logger.debug("Applying rule ID %s of type '%s'", rule.get("id"), rule_type)
        if rule_type not in self.handlers:
            raise NotImplementedError(f"Unsupported rule type: {rule_type}")
        return self.handlers[rule_type](rule)

    def apply_rules(self, rules: list[dict]) -> pd.DataFrame:
        self.fallback_notes = []

        for rule in rules:
            rule_id = rule.get("id", "unknown")
            logger.info("Applying rule: %s", rule_id)
            try:
                mask = self.apply_rule(rule)
                self.df[f"violation_{rule_id}"] = mask
                self.reports[rule_id] = {
                    "violations": int(mask.sum()),
                    "message": rule.get("message", "No message provided."),
                    "rule": rule.copy()
                }
                logger.debug("Rule %s flagged %d rows", rule_id, int(mask.sum()))
            except Exception as e:
                error_message = str(e)
                logger.error("Error applying rule %s: %s", rule_id, error_message)
                self.reports[rule_id] = {
                    "message": f"Error applying rule: {error_message}",
                    "rule": rule,
                }

                message = rule.get("message") or f"Failed to apply rule {rule_id}"
                self.fallback_notes.append(message)

        return self.df

    def get_violations_for_rule(self, rule_id: str, show: str = "all") -> pd.DataFrame:
        flag_col = f"violation_{rule_id}"
        logger.debug("Fetching violations for rule: %s", rule_id)

        if flag_col not in self.df.columns:
            raise ValueError(f"Violation column '{flag_col}' not found.")

        df_violations = self.df[self.df[flag_col]]

        if show == "rule":
            rule = self.reports.get(rule_id, {}).get("rule", {})
            columns = rule.get("columns") or [rule.get("column")]
            columns = [c for c in columns if c is not None and c in self.df.columns]
            return df_violations[columns + [flag_col]]

        return df_violations

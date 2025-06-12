from collections.abc import Callable

import pandas as pd

from data_quality_tool.config.logging_config import get_logger

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
        rule_id = rule.get("id", "<unknown>")
        logger.debug("Handling 'conditional' rule: %s", rule_id)

        cond = rule.get("condition", {})
        if_cond = cond.get("if", {})
        then_cond = cond.get("then", {})

        def parse_if_condition(if_block: dict) -> pd.Series:
            if isinstance(if_block, dict) and "column" in if_block and "equals" in if_block:
                col, val = if_block["column"], if_block["equals"]
            elif len(if_block) == 1:
                col, val = list(if_block.items())[0]
            else:
                raise ValueError(f"[{rule_id}] Invalid IF condition: {if_block}")

            if col not in self.df.columns:
                raise ValueError(f"[{rule_id}] IF column '{col}' not found in DataFrame")

            return self.df[col] == val

        def parse_then_condition(then_block: dict) -> pd.Series:
            if "column" in then_block:
                col = then_block["column"]
                if col not in self.df.columns:
                    raise ValueError(f"[{rule_id}] THEN column '{col}' not found in DataFrame")

                if then_block.get("not_null") is True:
                    return self.df[col].isnull()

                condition = then_block.get("condition", {})
            elif len(then_block) == 1:
                col, condition = list(then_block.items())[0]
                if col not in self.df.columns:
                    raise ValueError(f"[{rule_id}] THEN column '{col}' not found in DataFrame")

                if condition.get("not_null") is True:
                    return self.df[col].isnull()
            else:
                raise ValueError(f"[{rule_id}] Unsupported THEN format: {then_block}")

            if not condition:
                raise ValueError(f"[{rule_id}] Empty condition for THEN column '{col}'")

            mask = pd.Series(False, index=self.df.index)
            if "min" in condition:
                mask |= self.df[col] < condition["min"]
            if "max" in condition:
                mask |= self.df[col] > condition["max"]

            return mask

        mask_if = parse_if_condition(if_cond)
        mask_then = parse_then_condition(then_cond)

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

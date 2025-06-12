import pandas as pd

from data_quality_tool.config.logging_config import get_logger

logger = get_logger()


def inspect_rule_violations(df: pd.DataFrame, reports: dict):
    logger.info("Inspecting rule violations...")

    rule_ids = list(reports.keys())
    if not rule_ids:
        logger.info("No rules available for inspection.")
        return

    logger.info("Available rule IDs: %s", ", ".join(rule_ids))

    # Replace this with actual selection logic (e.g., CLI input, streamlit dropdown, etc.)
    logger.warning("Interactive input is disabled. Automatically inspecting the first rule.")
    rule_id = rule_ids[0]

    if rule_id not in reports:
        logger.error("Rule %s not found in the report.", rule_id)
        return

    try:
        rule = reports[rule_id].get("rule", {})
        flag_col = f"violation_{rule_id}"
        if flag_col not in df.columns:
            logger.warning("No violation column found for rule ID: %s", rule_id)
            return

        # Determine which columns to display
        cols = []
        if rule.get("type") == "conditional":
            cond = rule.get("condition", {})
            if_cond = cond.get("if", {})
            then_cond = cond.get("then", {})

            if isinstance(if_cond, dict):
                cols += [if_cond.get("column")] if "column" in if_cond else list(if_cond.keys())

            if isinstance(then_cond, dict):
                cols += [then_cond.get("column")] if "column" in then_cond else list(then_cond.keys())
        else:
            col = rule.get("column") or rule.get("columns")
            if isinstance(col, str):
                cols.append(col)
            elif isinstance(col, list):
                cols.extend(col)

        cols = list(set(c for c in cols if c in df.columns))

        violations = df[df[flag_col]]
        logger.info("Found %d violations for rule %s. Displaying first 5 rows.", len(violations), rule_id)
        logger.debug("\n%s", violations[cols + [flag_col]].head().to_string())  # Includes index now


    except Exception as e:
        logger.exception("Could not retrieve violations for rule %s: %s", rule_id, str(e))

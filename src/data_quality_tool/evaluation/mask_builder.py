import pandas as pd
from data_quality_tool.config.logging_config import get_logger
from .checker import DataQualityChecker  # Import the checker for standard DQ issues

logger = get_logger()


def build_violation_mask(df: pd.DataFrame, rule_reports: dict, note_results: dict) -> pd.DataFrame:
    original_columns = [col for col in df.columns if not col.startswith("violation_")]
    mask_df = pd.DataFrame(index=df.index, columns=original_columns)

    def merge_flags(existing, new_flag):
        if pd.isna(existing) or existing == "":
            return new_flag
        return f"{existing}, {new_flag}"

    logger.info("Building violation mask from rules...")
    for rule_id, report in rule_reports.items():
        if "violations" not in report:
            logger.warning("Skipping rule %s due to missing violations", rule_id)
            continue

        rule = report.get("rule", {})
        violation_col = f"violation_{rule_id}"
        if violation_col not in df.columns:
            logger.warning("Violation column %s not found in DataFrame", violation_col)
            continue

        colnames = []
        if rule.get("type") == "conditional":
            cond = rule.get("condition", {})
            if_cond = cond.get("if", {})
            then_cond = cond.get("then", {})

            if isinstance(if_cond, dict):
                colnames.append(if_cond.get("column")) if "column" in if_cond else colnames.extend(if_cond.keys())

            if isinstance(then_cond, dict):
                colnames.append(then_cond.get("column")) if "column" in then_cond else colnames.extend(then_cond.keys())
        else:
            col = rule.get("column") or rule.get("columns")
            colnames.extend([col]) if isinstance(col, str) else colnames.extend(col or [])

        colnames = list(set([c for c in colnames if c in df.columns]))
        violations = df[violation_col]

        for col in colnames:
            mask_df[col] = mask_df[col].astype(object)
            idx = violations[violations].index
            for i in idx:
                mask_df.at[i, col] = merge_flags(mask_df.at[i, col], rule_id)

    logger.info("Processing note results...")
    for note, result in note_results.items():
        if "error" in result:
            logger.warning("Skipping note check %s due to error: %s", note, result["error"])
            continue

        violations = result.get("violations_mask")
        if not isinstance(violations, pd.Series):
            logger.warning("Note %s has no valid violations mask", note)
            continue

        note_id = result.get("id", "NOTE")
        affected_cols = result.get("columns", [])
        if not affected_cols:
            inferred_col = violations.name
            if inferred_col and inferred_col in df.columns:
                affected_cols = [inferred_col]
                logger.info("Inferred affected column '%s' for note '%s' from violations mask.", inferred_col, note)
            else:
                logger.warning("No affected columns found or inferred for note '%s'. Skipping.", note)
                continue

        for col in affected_cols:
            if col not in df.columns:
                logger.warning("Column %s from note %s not found in DataFrame", col, note_id)
                continue
            mask_df[col] = mask_df[col].astype(object)
            idx = violations[violations].index
            for i in idx:
                mask_df.at[i, col] = merge_flags(mask_df.at[i, col], note_id)

    # ✅ Add Standard DQ Violations
    logger.info("Adding standard DQ violations...")
    dq_checker = DataQualityChecker(df)
    dq_checker.run_all_checks()
    std_dq_mask = dq_checker.generate_violation_mask()

    for col in mask_df.columns:
        mask_df[col] = mask_df[col].astype(object)
        for idx in mask_df.index:
            std_flag = std_dq_mask.at[idx, col]
            if std_flag:
                mask_df.at[idx, col] = merge_flags(mask_df.at[idx, col], std_flag)

    # Save Final Mask
    mask_df.to_csv("prediction_mask.csv", index=False)
    logger.info("Final combined violation mask saved to prediction_mask.csv")

    return mask_df

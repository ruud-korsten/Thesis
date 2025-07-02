import pandas as pd
from data_quality_tool.config.logging_config import get_logger
from data_quality_tool.dq_check.base_dq_check import DataQualityChecker
import numpy as np

logger = get_logger()

def build_violation_mask(df: pd.DataFrame, rule_reports: dict, note_results: dict, expected_dtypes=None) -> pd.DataFrame:
    original_columns = [col for col in df.columns if not col.startswith("violation_")]
    mask_df = pd.DataFrame(index=df.index, columns=original_columns).astype(object)

    def merge_flags(existing, new_flag):
        if pd.isna(existing) or existing == "":
            return new_flag
        return f"{existing}, {new_flag}"

    # === Rules ===
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
            for cond_part in [cond.get("if", {}), cond.get("then", {})]:
                if isinstance(cond_part, dict):
                    if "column" in cond_part:
                        colnames.append(cond_part["column"])
                    else:
                        colnames.extend(cond_part.keys())
        else:
            col = rule.get("column") or rule.get("columns")
            colnames.extend([col] if isinstance(col, str) else (col or []))

        colnames = list(set([c for c in colnames if c in df.columns]))
        violations = df[violation_col]

        for col in colnames:
            idx = violations[violations].index
            existing = mask_df.loc[idx, col].fillna("")
            updated = pd.Series(
                [merge_flags(val, rule_id) for i, val in existing.items()],
                index=existing.index
            )
            mask_df.loc[idx, col] = updated

    logger.info("Rule-based masking complete.")

    # === Notes ===
    logger.info("Processing note results...")
    for note, result in note_results.get("passed", {}).items():
        logger.info("Processing note: %s", note)

        violations = result.get("violations_mask")
        if not isinstance(violations, pd.Series):
            logger.warning("Note '%s' has no valid violations mask (type: %s)", note, type(violations))
            continue

        violations = violations.fillna(False).astype(bool)
        idx = violations[violations].index

        note_id = result.get("id", "NOTE")
        affected_cols = result.get("columns", [])

        if not affected_cols:
            # Fallback: try to infer from Series name
            inferred_col = violations.name
            if inferred_col and inferred_col in df.columns:
                affected_cols = [inferred_col]
                logger.info("Note '%s' - Inferred column from Series name: %s", note, inferred_col)
            else:
                logger.warning("Note '%s' - Could not determine affected columns. Skipping.", note)
                continue

        for col in affected_cols:
            if col not in df.columns:
                logger.warning("Note '%s' - Column '%s' not in DataFrame. Skipping.", note_id, col)
                continue

            existing = mask_df.loc[idx, col].fillna("")
            updated = pd.Series(
                [merge_flags(val, note_id) for val in existing],
                index=existing.index
            )
            mask_df.loc[idx, col] = updated

        logger.info("Note '%s' processing complete.", note)


    # === Standard DQ Violations ===
    logger.info("Adding standard DQ violations...")
    dq_checker = DataQualityChecker(df, expected_dtypes=expected_dtypes)
    dq_checker.run_all_checks()
    std_dq_mask = dq_checker.generate_violation_mask()

    for col in mask_df.columns:
        std_flags = std_dq_mask[col]
        idx = std_flags[std_flags != ""].index
        existing = mask_df.loc[idx, col].fillna("")
        updated = pd.Series(
            [merge_flags(val, std_flags.loc[i]) for i, val in existing.items()],
            index=existing.index
        )
        mask_df.loc[idx, col] = updated

    logger.info("Standard DQ masking complete.")
    logger.info("Final mask shape: %s", mask_df.shape)

    output_path = "artifacts/prediction_mask.csv"
    mask_df.to_csv(output_path, index=False)
    logger.info("Final combined violation mask saved to %s", output_path)

    return mask_df

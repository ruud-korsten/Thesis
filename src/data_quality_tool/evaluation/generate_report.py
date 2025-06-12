import pandas as pd
from data_quality_tool.dq_check.base_dq_check import DataQualityChecker
from data_quality_tool.config.logging_config import get_logger

logger = get_logger()


def generate_dq_report(df, rule_reports, note_results, output_path="artifacts/dq_report.md", expected_dtypes=None):

    logger.info("Starting data quality report generation.")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            # === 1. Dataset Overview ===
            logger.debug("Writing dataset overview section.")
            f.write("# Data Quality Report\n\n")

            f.write("## Dataset Overview\n")
            f.write(f"- **Shape**: `{df.shape[0]} rows × {df.shape[1]} columns`\n")
            f.write(f"- **Columns**: `{', '.join(df.columns)}`\n\n")

            f.write("### Sample Rows (first 5)\n")
            f.write(df.head(5).to_markdown(index=False))
            f.write("\n\n")

            f.write("### Summary Statistics\n")
            summary = df.describe(include='all').transpose()
            f.write(summary.to_markdown())
            f.write("\n\n")

            # === 2. Standard DQ Checks ===
            logger.debug("Running standard DQ checks.")
            f.write("## Standard DQ Issues\n")
            dq_checker = DataQualityChecker(df, expected_dtypes=expected_dtypes)
            dq_results = dq_checker.run_all_checks()
            for issue_type, issue_df in dq_results.items():
                f.write(f"### {issue_type.title()}:\n")

                if issue_df.empty:
                    f.write("_No issues found._\n\n")
                    logger.debug(f"No issues found for check: {issue_type}")
                    continue

                display_df = issue_df.head(5) if len(issue_df) > 10 else issue_df
                f.write(display_df.to_markdown(index=True))

                if len(issue_df) > 5:
                    f.write(f"\n_...showing first 5 of {len(issue_df)} rows._\n")

                f.write("\n\n")
                logger.info(f"Issues found in {issue_type}: {len(issue_df)}")

            # === 3. Rule Violations ===
            logger.debug("Processing rule violations.")
            f.write("## Rules & Notes Summary\n")
            f.write("### Rule Violations\n")
            rule_rows = []
            for rule_id, report in rule_reports.items():
                message = report.get("message", "")
                if "error" in message.lower():
                    logger.warning(f"Skipping rule {rule_id} due to error: {message}")
                    continue
                count = report.get("violations", 0)
                rule_rows.append((rule_id, count, message))

            if rule_rows:
                rule_df = pd.DataFrame(rule_rows, columns=["Rule ID", "Violations", "Description"])
                f.write(rule_df.to_markdown(index=False))
                logger.info(f"{len(rule_rows)} rules reported with violations.")
            else:
                f.write("_No rule violations reported (excluding errored rules)._")
                logger.info("No valid rule violations to report.")
            f.write("\n\n")

            # === 4. Note Violations ===
            logger.debug("Processing note violations.")
            f.write("### Note Violations\n")
            note_rows = []

            for note, result in note_results.get("passed", {}).items():
                count = result.get("violations", 0)
                note_rows.append((note, count, result.get("function_name", "")))

            for note, result in note_results.get("failed", {}).items():
                error = result.get("error", "Unknown error")
                note_rows.append((note, "ERROR", error))
                logger.error(f"Note check failed: {note} - {error}")

            note_df = pd.DataFrame(note_rows, columns=["Note", "Violations", "Function / Error"])
            f.write(note_df.to_markdown(index=False))
            f.write("\n\n")

        logger.info(f"Data quality report successfully saved to {output_path}")

    except Exception as e:
        logger.exception(f"Failed to generate DQ report: {e}")

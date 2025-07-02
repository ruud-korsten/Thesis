import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import glob
import shutil
from pathlib import Path
from data_quality_tool.config.logging_config import get_logger
from data_quality_tool.data.dataset_selection import load_dataset
from data_quality_tool.domain.domain_extractor import DomainExtractor
from data_quality_tool.domain.rule_note_parser import RuleParser
from data_quality_tool.dq_check.base_dq_check import DataQualityChecker
from data_quality_tool.evaluation.accuracy_evaluation import (
    evaluate_from_files,
    get_ground_truth_mask_path,
)
from data_quality_tool.evaluation.generate_report import generate_dq_report
from data_quality_tool.evaluation.mask_builder import build_violation_mask
from data_quality_tool.notes.note_engine import NoteEngine
from data_quality_tool.notes.note_evaluator import NoteEvaluator
from data_quality_tool.notes.note_inspector import inspect_note_violations
from data_quality_tool.rules.rule_engine import RuleExecutor
from data_quality_tool.rules.rule_inspector import inspect_rule_violations
from data_quality_tool.evaluation.save_run import save_run_snapshot
from data_quality_tool.evaluation.summary_writer import save_rule_note_summary
from data_quality_tool.notes.notes_feedback_loop import NoteFeedbackLoop
from data_quality_tool.evaluation.final_validation import FinalValidator


# === Load Configuration ===
load_dotenv()

logger = get_logger()


# Configurable Parameters
DATASET_NAME = os.getenv("DATASET_NAME")
DATASET_DIRTY = os.getenv("DATASET_DIRTY", "False").lower() == "true"
GROUND_TRUTH_MASK = get_ground_truth_mask_path(DATASET_NAME, dirty=DATASET_DIRTY)
DOMAIN_SAVE_DIR = os.getenv("DOMAIN_SAVE_DIR", "domain_knowledge/llm")
RULES_CACHE_PATH = os.getenv("RULES_CACHE_PATH", "artifacts/dq_rules.json")
RULES_RCI = os.getenv("RULES_RCI", "False").lower() == "true"
NOTES_REFRESH = os.getenv("NOTES_REFRESH", "False").lower() == "true"
NOTES_RCI = os.getenv("NOTES_RCI", "False").lower() == "true"
PREDICTION_MASK = os.getenv("PREDICTION_MASK_PATH", "artifacts/prediction_mask.csv")
DOMAIN_EXTRACTOR = os.getenv("DOMAIN_EXTRACTOR", "False").lower() == "true"
LAST_DATASET_FILE = "artifacts/last_dataset_used.txt"
ACCURACY_PATH="artifacts/accuracy.json"
FINAL_VALIDATION = os.getenv("FINAL_VALIDATION", "False").lower() == "true"
SMALL_DATASET = os.getenv("SMALL_DATASET", "False").lower() == "true"
KEEP_ALL_RUNS = os.getenv("KEEP_ALL_RUNS", "True").lower() == "true"
MULTI_DOMAIN_EXTRACTION = os.getenv("MULTI_DOMAIN_EXTRACTION", "False").lower() == "true"

def should_force_refresh_based_on_dataset(dataset_name: str, dirty: bool) -> bool:
    """
    Force refresh if the dataset name or dirty flag has changed since the last run.
    """
    user_refresh = os.getenv("RULES_FORCE_REFRESH", "False").lower() == "true"
    if user_refresh:
        logger.info("RULES_FORCE_REFRESH is set to True. Forcing refresh by user request.")
        return True

    current_dataset_id = f"{dataset_name}_dirty={dirty}"

    if os.path.exists(LAST_DATASET_FILE):
        with open(LAST_DATASET_FILE) as f:
            last_dataset_id = f.read().strip()
        if last_dataset_id == current_dataset_id:
            logger.info("Same dataset and dirty flag detected. Using cached results.")
            return False
        else:
            logger.info("Dataset or dirty flag changed. Forcing refresh.")
    else:
        logger.info("No record of previously used dataset. Forcing refresh.")

    # Store current dataset info for future runs
    with open(LAST_DATASET_FILE, "w") as f:
        f.write(current_dataset_id)

    return True


def run_basic_dq(df, expected_dtypes=None):
    dq = DataQualityChecker(df, expected_dtypes=expected_dtypes)
    reports = dq.run_all_checks()

    if 'missing' in reports:
        logger.info("Missing Values:\n%s", reports['missing'].head().to_string(index=True))

    if 'duplicates' in reports:
        logger.info("Duplicate Summary:\n%s", reports['duplicates'].to_string(index=False))

    if 'schema_mismatches' in reports:
        logger.info("Schema Mismatches:\n%s", reports['schema_mismatches'].to_string(index=True))

    return reports


def run_rule_engine(df: pd.DataFrame, rules: list[dict]) -> tuple[pd.DataFrame, RuleExecutor]:
    executor = RuleExecutor(df)
    result_df = executor.apply_rules(rules)

    logger.info("Rule Violation Summary:")
    for rule_id, report in executor.reports.items():
        if "error" in report.get("message", "").lower():
            logger.error("%s: ERROR - %s", rule_id, report["message"])
        else:
            logger.info("%s: %d violations - %s", rule_id, report["violations"], report["message"])

    return result_df, executor

def run_and_evaluate_note_engine(notes: list[str], df: pd.DataFrame, use_rci: bool = None) -> tuple[dict, list[dict]]:
    if not notes:
        logger.info("No domain notes found to process.")
        return {}, []

    if use_rci is None:
        use_rci = os.getenv("NOTES_RCI", "False").lower() == "true"

    logger.info(
        "Running Note Engine on domain insights with NOTES_REFRESH=%s and NOTES_RCI=%s",
        NOTES_REFRESH, use_rci
    )

    note_engine = NoteEngine()
    note_functions, note_usage_stats = note_engine.run(
        notes=notes,
        df=df,
        cache_path="artifacts/note_functions.json",
        force_refresh=NOTES_REFRESH,
        use_rci=use_rci
    )

    logger.info("Evaluating generated data quality checks...")
    evaluator = NoteEvaluator(df)
    results = evaluator.evaluate(note_functions)

    passed = results.get("passed", {})
    failed = results.get("failed", {})

    logger.info("Summary: %d passed, %d failed", len(passed), len(failed))

    for note, result in passed.items():
        logger.info("\n---\nNote: %s", note)
        logger.info("Function: %s", result["function_name"])
        logger.info("Violations flagged: %d", result["violations"])
        logger.debug("Code:\n%s", result["code"])

    for note, result in failed.items():
        logger.warning("\n---\nNote FAILED: %s", note)
        logger.warning("Error: %s", result["error"])
        logger.debug("Code:\n%s", result["code"])

    return results, note_usage_stats

def main():
    save_path = f"{DOMAIN_SAVE_DIR}/{DATASET_NAME}.txt"

    logger.info("\n" + "=" * 80)
    logger.info("Starting new Data Quality run")
    logger.info("=" * 30 + " [1. DATA LOADING & INJECTION] " + "=" * 30)

    df, expected_dtypes, domain_path = load_dataset(DATASET_NAME, dirty=DATASET_DIRTY)
    if SMALL_DATASET:
        df = df.head(10000)

    if DATASET_NAME == "energy":
        # Set 3 random indexes where usage < 10000 to a high value
        mask = df["usage"] < 10000
        eligible_indices = df[mask].sample(3, random_state=42).index
        df.loc[eligible_indices, "usage"] = 1e8  # Or any high value

        # Make 4 random usage cells empty (NaN)
        empty_indices = df.sample(4, random_state=43).index
        df.loc[empty_indices, "usage"] = np.nan

        # Inject negative values in both usage and return columns, in the same 2 rows
        neg_indices = df.sample(2, random_state=44).index
        df.loc[neg_indices, "usage"] = -1 * np.abs(df.loc[neg_indices, "usage"].fillna(1))
        df.loc[neg_indices, "return"] = np.random.uniform(0, 5) * -1

        # Set 3 random date values to a future date
        future_date = pd.Timestamp("2026-01-01", tz="UTC")
        future_date_indices = df.sample(3, random_state=45).index
        df.loc[future_date_indices, "datetime_recorded"] = future_date

        # Make datetime_recorded timezone naive
        df["datetime_recorded"] = pd.to_datetime(df["datetime_recorded"]).dt.tz_localize(None)

    logger.info("Dataset Loaded:\n%s", df.head().to_string(index=False))
    logger.info("Dataset Summary Stats:\n%s", df.describe(include='all').transpose().to_string())

    logger.info("=" * 30 + " [2. STANDARD DQ CHECKS] " + "=" * 30)
    run_basic_dq(df, expected_dtypes=expected_dtypes)

    all_usage = []

    if DOMAIN_EXTRACTOR:
        logger.info("=" * 30 + " [DOMAIN EXTRACTION] " + "=" * 30)

        if MULTI_DOMAIN_EXTRACTION:
            from data_quality_tool.domain.domain_extractor import DomainExtractorMultiple

            # Load additional datasets for this domain
            domain_dir = f"data/{DATASET_NAME}/raw"
            context_dfs = {}
            if os.path.exists(domain_dir):
                for file in os.listdir(domain_dir):
                    if file.endswith(".xlsx"):
                        try:
                            df_extra = pd.read_excel(os.path.join(domain_dir, file))
                            key = os.path.splitext(file)[0]
                            context_dfs[key] = df_extra
                            logger.info("Loaded context dataset: %s", file)
                        except Exception as e:
                            logger.warning("Failed to load %s for multi-domain extraction: %s", file, e)

            # Run domain extraction with primary + context
            extractor = DomainExtractorMultiple()
            domain_output = extractor.extract_domain_knowledge(primary_df=df, context_dfs=context_dfs,
                                                               save_path=save_path)

        else:
            extractor = DomainExtractor()
            domain_output = extractor.extract_domain_knowledge(df, save_path=save_path)

        domain_path = save_path

        if isinstance(domain_output, dict) and "usage" in domain_output:
            all_usage.append({"stage": "domain_extraction", **domain_output["usage"]})

        logger.info("Extracted Domain Knowledge:\n%s", domain_output.get("response", "No response received."))

    logger.info("=" * 30 + " [3. RULES & NOTES] " + "=" * 30)
    parser = RuleParser()
    FORCE_REFRESH_DYNAMIC = should_force_refresh_based_on_dataset(DATASET_NAME, DATASET_DIRTY)

    rules, notes, rule_usage = parser.parse_rules(
        rules_path=domain_path,
        cache_path=RULES_CACHE_PATH,
        force_refresh=FORCE_REFRESH_DYNAMIC,
        df=df,
        use_rci=RULES_RCI
    )

    if RULES_RCI:
        for substage, usage in rule_usage.items():
            all_usage.append({
                "stage": f"rule_parsing/{substage}",
                **usage
            })
    else:
        all_usage.append({
            "stage": "rule_parsing",
            **rule_usage
        })

    logger.info("Rule Extraction Complete. Extracted %d rules.", len(rules))
    logger.info("See %s for full rule output.", RULES_CACHE_PATH)

    result_df, executor = run_rule_engine(df, rules)
    inspect_rule_violations(result_df, executor.reports)

    logger.info("Fallback Notes:\n%s", executor.fallback_notes)
    all_notes = notes + executor.fallback_notes
    logger.info("Total Notes to Evaluate: %d", len(all_notes))

    note_results, note_usage_stats = run_and_evaluate_note_engine(all_notes, df)
    for note, usage in note_usage_stats.items():
        print(note,usage)
        for substage, subusage in usage.items():
            all_usage.append({
                "stage": f"note_engine/{substage}",
                "note": note,
                **subusage
            })


    feedback = NoteFeedbackLoop(df=df)
    repaired_notes, retry_usage = feedback.retry_failed_notes(note_results)
    for note, result in repaired_notes.items():
        if "error" not in result:
            note_results["passed"][note] = result
            note_results["failed"].pop(note, None)
        else:
            note_results["failed"][note] = result

    for note, usage in retry_usage.items():
        all_usage.append({
            "stage": "note_retry",
            "note": note,
            **usage
        })

    inspect_note_violations(df, note_results)

    logger.info("=" * 30 + " [5. EVALUATION] " + "=" * 30)
    generate_dq_report(df, executor.reports, note_results, expected_dtypes=expected_dtypes)

    mask_df = build_violation_mask(result_df, executor.reports, note_results, expected_dtypes=expected_dtypes)
    logger.info("Mask built successfully. shape: %s", mask_df.shape)
    accuracy = None
    if DATASET_DIRTY:
        accuracy = evaluate_from_files(GROUND_TRUTH_MASK, PREDICTION_MASK)

    logger.info("=" * 30 + " [6. SNAPSHOT EXPORT] " + "=" * 30)
    save_rule_note_summary(executor.reports, note_results)

    final_validation = None
    if FINAL_VALIDATION:
        validator = FinalValidator(run_dir="artifacts", dataset=df)
        final_validation, validation_usage = validator.validate()
        all_usage.append({"stage": "final_validation", **validation_usage})
        logger.info("Validation Feedback:\n%s", final_validation)

    run_dir = save_run_snapshot(
        dataset_name=DATASET_NAME,
        pred_mask=mask_df,
        true_mask_path=GROUND_TRUTH_MASK,
        dataset=df,
        run_accuracy=accuracy if accuracy else None,
        final_validation=final_validation,
        domain_file_dir=domain_path
    )

    # Ensure Windows path separator compatibility
    run_dir_path = Path(run_dir)
    parent_run_folder = run_dir_path.parent
    latest_run_name = run_dir_path.name

    if not KEEP_ALL_RUNS:
        logger.info("SAVE_ALL_RUNS is False — removing older runs in: %s", parent_run_folder)
        for subdir in parent_run_folder.iterdir():
            if subdir.is_dir() and subdir.name != latest_run_name:
                try:
                    shutil.rmtree(subdir)
                    logger.info("Deleted old run directory: %s", subdir)
                except Exception as e:
                    logger.warning("Could not delete directory %s: %s", subdir, e)
    else:
        logger.info("KEEP_ALL_RUNS is True — keeping all run folders.")

    # Optionally log or save usage summary
    logger.debug("All usage entries:")
    for i, u in enumerate(all_usage):
        logger.debug("  %d: type=%s value=%s", i, type(u), repr(u))

    logger.info("Token and cost summary:")

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_cost = 0.0

    for usage in all_usage:
        if isinstance(usage, dict) and "total_tokens" in usage and "estimated_cost" in usage:
            logger.info(
                "%s: %d prompt, %d completion, %d total tokens, $%.5f",
                usage.get("stage", "unknown"),
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                usage["total_tokens"],
                usage["estimated_cost"]
            )
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
            total_tokens += usage["total_tokens"]
            total_cost += usage["estimated_cost"]
        else:
            logger.warning("Skipping malformed usage entry: %s", usage)

    logger.info(
        "TOTAL USAGE: %d prompt, %d completion, %d total tokens, $%.5f",
        total_prompt_tokens,
        total_completion_tokens,
        total_tokens,
        total_cost
    )


if __name__ == "__main__":
    main()

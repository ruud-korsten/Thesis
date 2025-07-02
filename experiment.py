import os
import time
import pandas as pd
from dotenv import load_dotenv
from collections import defaultdict
import csv
from statistics import mean, stdev
from main import run_basic_dq, run_rule_engine, run_and_evaluate_note_engine
from data_quality_tool.config.logging_config import get_logger
from data_quality_tool.data.dataset_selection import load_dataset
from data_quality_tool.domain.domain_extractor import DomainExtractor
from data_quality_tool.domain.rule_note_parser import RuleParser
from data_quality_tool.dq_check.base_dq_check import DataQualityChecker
from data_quality_tool.evaluation.accuracy_evaluation import evaluate_from_files, get_ground_truth_mask_path, evaluate_dq_performance
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

PER_RUN_METRICS_PATH = "experiment_results/runs"
AGGREGATE_METRICS_PATH = "experiment_results/aggregates"
os.makedirs(os.path.dirname(PER_RUN_METRICS_PATH), exist_ok=True)

DOMAIN_SAVE_DIR = os.getenv("DOMAIN_SAVE_DIR", "domain_knowledge/llm")

load_dotenv()
logger = get_logger(name="experiment_runner", log_file="logs/experiment.log")
#"health", "retail", "wallmart"
datasets = ["sensor"]
error_types = ["ERROR_MISSING","ERROR_TYPE_MISMATCH","ERROR_OUTLIERS","ERROR_DUPLICATES", "ERROR_OUTLIERS"]
error_levels = [0.01, 0.1]
models = ["gpt-4o", "deepseek-chat", "mistral-medium-2505-ruud", "grok-3", "grok-3-mini"]
#"gpt-4o-mini"
feature_configs = [
    {"name": "NoFeatures", "RCI": False, "FinalValidation": False, "DomainExtraction": False},
    #{"name": "RCIOnly", "RCI": True, "FinalValidation": False, "DomainExtraction": False},
    #{"name": "DomainOnly", "RCI": False, "FinalValidation": False, "DomainExtraction": True},
    #{"name": "Domain+RCI", "RCI": True, "FinalValidation": False, "DomainExtraction": True},
]

size_tiers = {"Medium": 10000, "Large": 100000}
sliding_steps = 5

token_stats = defaultdict(float)

per_run_results = []
aggregate_results = defaultdict(list)

for dataset in datasets:
    if dataset == "sensor":
        os.environ["DISABLE_MISSING"] = "True"
    else:
        os.environ["DISABLE_MISSING"] = "False"
    save_path = f"{DOMAIN_SAVE_DIR}/{dataset}.txt"
    for size_label, total_rows in size_tiers.items():
        os.environ["MAX_EXCEL_ROWS"] = str(total_rows)
        for model in models:
            os.environ["OPENAI_MODEL"] = model
            for feature_conf in feature_configs:
                os.environ["RULES_RCI"] = str(feature_conf["RCI"])
                os.environ["NOTES_RCI"] = str(feature_conf["RCI"])
                for error_level in error_levels:
                    for error_type in error_types:
                        os.environ[error_type] = str(error_level)

                    df_full, expected_dtypes, domain_path = load_dataset(dataset, dirty=True)
                    if len(df_full) < total_rows:
                        logger.warning("Dataset %s smaller than required rows (%d < %d). Skipping.", dataset, len(df_full), total_rows)
                        continue

                    step = total_rows // sliding_steps
                    for run_index in range(sliding_steps):
                        row_limit = step * (run_index + 1)
                        df = df_full.iloc[:row_limit].copy()
                        run_label = f"{dataset} | size={size_label} | model={model} | features={feature_conf['name']} | error={error_level} | rows={row_limit}"

                        logger.info("=== %s ===", run_label)
                        all_usage = []
                        start_time = time.time()

                        try:
                            run_basic_dq(df, expected_dtypes=expected_dtypes)

                            if feature_conf["DomainExtraction"]:
                                extractor = DomainExtractor(model=model)
                                domain_output = extractor.extract_domain_knowledge(df, save_path=save_path)
                                domain_path=save_path
                                if "usage" in domain_output:
                                    all_usage.append({"stage": "domain_extraction", **domain_output["usage"]})

                            parser = RuleParser(model=model)
                            rules, notes, rule_usage = parser.parse_rules(
                                rules_path=domain_path,
                                cache_path="artifacts/dq_rules.json",
                                force_refresh=True,
                                df=df,
                                use_rci=feature_conf["RCI"]
                            )
                            if feature_conf["RCI"]:
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

                            result_df, executor = run_rule_engine(df, rules)
                            inspect_rule_violations(result_df, executor.reports)

                            all_notes = notes + executor.fallback_notes
                            note_engine = NoteEngine(model=model)
                            note_results, note_usage_stats = run_and_evaluate_note_engine(all_notes, df, use_rci=feature_conf["RCI"])
                            for note, usage in note_usage_stats.items():
                                print(note, usage)
                                for substage, subusage in usage.items():
                                    all_usage.append({
                                        "stage": f"note_engine/{substage}",
                                        "note": note,
                                        **subusage
                                    })

                            feedback = NoteFeedbackLoop(df=df, model=model)
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

                            generate_dq_report(df, executor.reports, note_results, expected_dtypes=expected_dtypes)
                            mask_df = build_violation_mask(result_df, executor.reports, note_results, expected_dtypes=expected_dtypes)

                            acc = None
                            gt_mask_path = get_ground_truth_mask_path(dataset, dirty=True)
                            logger.debug("Using ground truth from: %s", gt_mask_path)
                            gt_mask_df = pd.read_excel(gt_mask_path).iloc[:row_limit].copy()
                            gt_cleaned = gt_mask_df.replace("noop", pd.NA)

                            if gt_cleaned.dropna(how="all").empty:
                                logger.info("True mask is clean, evaluating for false positives only")
                            acc = evaluate_dq_performance(gt_mask_df, mask_df)
                            logger.info("Acc: %s", acc)

                            final_val = None
                            if feature_conf["FinalValidation"]:
                                validator = FinalValidator(run_dir="artifacts", dataset=df)
                                final_val, validation_usage = validator.validate()
                                all_usage.append({"stage": "final_validation", **validation_usage})

                            save_rule_note_summary(executor.reports, note_results)
                            save_run_snapshot(
                                dataset_name=dataset,
                                pred_mask=mask_df,
                                true_mask_path=gt_mask_path if gt_mask_df is not None else None,
                                dataset=df,
                                run_accuracy=acc,
                                final_validation=final_val
                            )

                            elapsed_time = time.time() - start_time
                            logger.info("Run time: %.2f seconds", elapsed_time)
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

                            logger.info("Done with %s", run_label)

                            # === Store per-run metrics ===
                            metrics = acc.get("overall_metrics", {}) if acc else {}

                            run_metrics = {
                                "dataset": dataset,
                                "size": size_label,
                                "model": model,
                                "features": feature_conf["name"],
                                "error_level": error_level,
                                "run_index": run_index + 1,
                                "accuracy": metrics.get("accuracy"),
                                "f1": metrics.get("f1_score"),
                                "precision": metrics.get("precision"),
                                "recall": metrics.get("recall"),
                                "runtime_seconds": elapsed_time,
                                "tokens_total": total_tokens,
                                "tokens_prompt": total_prompt_tokens,
                                "tokens_completion": total_completion_tokens,
                                "cost_usd": total_cost
                            }
                            per_run_results.append(run_metrics)



                        except Exception as e:
                            logger.exception("Failed run: %s", run_label)
                    # === Save metrics for this configuration ===

                    # Clean model name for folder safety
                    model_name = model.replace('/', '_')

                    # Define output folders per model and dataset
                    run_output_dir = os.path.join(PER_RUN_METRICS_PATH, model_name, dataset)
                    agg_output_dir = os.path.join(AGGREGATE_METRICS_PATH, model_name, dataset)
                    os.makedirs(run_output_dir, exist_ok=True)
                    os.makedirs(agg_output_dir, exist_ok=True)

                    # Build config ID
                    config_id = f"{model_name}_{dataset}_{size_label}_{feature_conf['name']}_err{error_level}"

                    # Filter runs for this config
                    config_run_data = [
                        r for r in per_run_results
                        if r["dataset"] == dataset and r["size"] == size_label
                           and r["model"] == model and r["features"] == feature_conf["name"]
                           and r["error_level"] == error_level
                    ]

                    # Save per-run metrics
                    run_path = os.path.join(run_output_dir, f"{config_id}_runs.csv")
                    pd.DataFrame(config_run_data).to_csv(run_path, index=False)

                    # Save aggregate metrics
                    if config_run_data:
                        f1_scores = [r["f1"] for r in config_run_data if r["f1"] is not None]
                        acc_scores = [r["accuracy"] for r in config_run_data if r["accuracy"] is not None]

                        agg = {
                            "dataset": dataset,
                            "size": size_label,
                            "model": model,
                            "features": feature_conf["name"],
                            "error_level": error_level,
                            "mean_accuracy": mean(acc_scores) if acc_scores else None,
                            "mean_f1": mean(f1_scores) if f1_scores else None,
                            "std_f1": stdev(f1_scores) if len(f1_scores) > 1 else 0.0,
                            "mean_latency": mean(r["runtime_seconds"] for r in config_run_data),
                            "mean_tokens": mean(r["tokens_total"] for r in config_run_data),
                            "runs_count": len(config_run_data)
                        }

                        agg_path = os.path.join(agg_output_dir, f"{config_id}_agg.csv")
                        pd.DataFrame([agg]).to_csv(agg_path, index=False)
                        logger.info("Saved metrics for config %s", config_id)

logger.info("All experiments complete")

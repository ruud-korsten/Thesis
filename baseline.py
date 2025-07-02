import os
import time
import pandas as pd
from dotenv import load_dotenv
from statistics import mean, stdev
from collections import defaultdict
from main import run_basic_dq
from data_quality_tool.config.logging_config import get_logger
from data_quality_tool.data.dataset_selection import load_dataset
from data_quality_tool.evaluation.save_run import save_run_snapshot
from data_quality_tool.evaluation.accuracy_evaluation import evaluate_dq_performance, get_ground_truth_mask_path
from data_quality_tool.evaluation.mask_builder import build_violation_mask

load_dotenv()
logger = get_logger(name="baseline_runner", log_file="logs/baseline_experiment.log")

PER_RUN_METRICS_PATH = "experiment_results/runs"
AGGREGATE_METRICS_PATH = "experiment_results/aggregates"
os.makedirs(os.path.dirname(PER_RUN_METRICS_PATH), exist_ok=True)

datasets = ["health", "retail", "wallmart", "sensor"]
error_types = ["ERROR_MISSING", "ERROR_TYPE_MISMATCH", "ERROR_OUTLIERS", "ERROR_DUPLICATES"]
error_levels = [0.01, 0.1]
size_tiers = {"Medium": 10000, "Large": 100000}
sliding_steps = 5

per_run_results = []

for dataset in datasets:
    for size_label, total_rows in size_tiers.items():
        os.environ["MAX_EXCEL_ROWS"] = str(total_rows)
        for error_level in error_levels:
            for error_type in error_types:
                os.environ[error_type] = str(error_level)

            df_full, expected_dtypes, _ = load_dataset(dataset, dirty=True)
            if len(df_full) < total_rows:
                logger.warning("Dataset %s smaller than required rows (%d < %d). Skipping.", dataset, len(df_full), total_rows)
                continue

            step = total_rows // sliding_steps
            for run_index in range(sliding_steps):
                row_limit = step * (run_index + 1)
                df = df_full.iloc[:row_limit].copy()
                run_label = f"{dataset} | size={size_label} | error={error_level} | rows={row_limit}"

                logger.info("=== %s ===", run_label)
                start_time = time.time()

                try:
                    dq_reports = run_basic_dq(df, expected_dtypes=expected_dtypes)
                    mask_df = build_violation_mask(df, rule_reports=dq_reports, note_results={},
                                                   expected_dtypes=expected_dtypes)
                    gt_mask_path = get_ground_truth_mask_path(dataset, dirty=True)
                    gt_df = pd.read_excel(gt_mask_path).iloc[:row_limit].copy()
                    acc = evaluate_dq_performance(gt_df, mask_df)

                    elapsed_time = time.time() - start_time
                    logger.info("Run time: %.2f seconds", elapsed_time)

                    metrics = acc.get("overall_metrics", {}) if acc else {}
                    run_metrics = {
                        "dataset": dataset,
                        "size": size_label,
                        "model": "baseline",
                        "features": "NoGenAI",
                        "error_level": error_level,
                        "run_index": run_index + 1,
                        "accuracy": metrics.get("accuracy"),
                        "f1": metrics.get("f1_score"),
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "runtime_seconds": elapsed_time,
                        "tokens_total": 0,
                        "tokens_prompt": 0,
                        "tokens_completion": 0,
                        "cost_usd": 0.0
                    }
                    per_run_results.append(run_metrics)

                except Exception as e:
                    logger.exception("Failed run: %s", run_label)

        # === Save metrics ===
        run_output_dir = os.path.join(PER_RUN_METRICS_PATH, "baseline", dataset)
        agg_output_dir = os.path.join(AGGREGATE_METRICS_PATH, "baseline", dataset)
        os.makedirs(run_output_dir, exist_ok=True)
        os.makedirs(agg_output_dir, exist_ok=True)

        for error_level in error_levels:
            config_id = f"baseline_{dataset}_{size_label}_NoGenAI_err{error_level}"

            config_run_data = [
                r for r in per_run_results
                if r["dataset"] == dataset and r["size"] == size_label
                and r["error_level"] == error_level and r["model"] == "baseline"
            ]

            # Save per-run
            run_path = os.path.join(run_output_dir, f"{config_id}_runs.csv")
            pd.DataFrame(config_run_data).to_csv(run_path, index=False)

            if config_run_data:
                from statistics import mean, stdev
                f1_scores = [r["f1"] for r in config_run_data if r["f1"] is not None]
                acc_scores = [r["accuracy"] for r in config_run_data if r["accuracy"] is not None]

                agg = {
                    "dataset": dataset,
                    "size": size_label,
                    "model": "baseline",
                    "features": "NoGenAI",
                    "error_level": error_level,
                    "mean_accuracy": mean(acc_scores) if acc_scores else None,
                    "mean_f1": mean(f1_scores) if f1_scores else None,
                    "std_f1": stdev(f1_scores) if len(f1_scores) > 1 else 0.0,
                    "mean_latency": mean(r["runtime_seconds"] for r in config_run_data),
                    "mean_tokens": 0,
                    "runs_count": len(config_run_data)
                }

                agg_path = os.path.join(agg_output_dir, f"{config_id}_agg.csv")
                pd.DataFrame([agg]).to_csv(agg_path, index=False)
                logger.info("Saved baseline metrics for config %s", config_id)

logger.info("All baseline experiments complete.")

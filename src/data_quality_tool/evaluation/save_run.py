import os
import shutil
import json
from datetime import datetime
import pandas as pd
from data_quality_tool.config.logging_config import get_logger

logger = get_logger()

def save_run_snapshot(
    dataset_name: str,
    pred_mask: pd.DataFrame,
    true_mask_path: str,
    dataset: pd.DataFrame,
    rules_path="artifacts/dq_rules.json",
    notes_path="artifacts/dq_notes.json",
    rules_summary_path="artifacts/rule_summary.csv",
    notes_summary_path="artifacts/note_summary.csv",
    note_functions_path="artifacts/note_functions.json",
    domain_file_dir="domain_knowledge",
    output_dir="artifacts/runs",
    accuracy_path="artifacts/accuracy.json",
    run_accuracy=None
):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(output_dir, dataset_name, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    logger.info("Saving run snapshot for dataset '%s' to %s", dataset_name, run_dir)

    # 1. Save prediction mask
    pred_path = os.path.join(run_dir, "prediction_mask.csv")
    pred_mask.to_csv(pred_path, index=False)
    logger.debug("Saved prediction mask to %s", pred_path)

    # 1.1. Save summaries
    rules_summary_dest = os.path.join(run_dir, "rule_summary.csv")
    notes_summary_dest = os.path.join(run_dir, "note_summary.csv")
    shutil.copy(rules_summary_path, rules_summary_dest)
    shutil.copy(notes_summary_path, notes_summary_dest)
    logger.debug("Copied rule and note summaries to %s and %s", rules_summary_dest, notes_summary_dest)

    # 2. Copy ground truth and dataset
    if run_accuracy:
        gt_dest = os.path.join(run_dir, "ground_truth_mask.xlsx")
        shutil.copy(true_mask_path, gt_dest)
        logger.debug("Copied ground truth mask to %s", gt_dest)

    dataset_dest = os.path.join(run_dir, "dataset.csv")
    dataset.to_csv(dataset_dest)
    logger.debug("Copied dataset to %s", dataset_dest)

    # 3. Copy rule/note artifacts
    for file, name in [(rules_path, "rules.json"), (notes_path, "notes.json"), (note_functions_path, "note_functions.json")]:
        dest = os.path.join(run_dir, name)
        if os.path.exists(file):
            shutil.copy(file, dest)
            logger.debug("Copied %s to %s", file, dest)
        else:
            logger.warning("Artifact not found: %s", file)

    # 4. Copy domain file
    domain_file = os.path.join(domain_file_dir, f"{dataset_name}.txt")
    domain_dest = os.path.join(run_dir, "domain.txt")
    if os.path.exists(domain_file):
        shutil.copy(domain_file, domain_dest)
        logger.debug("Copied domain file to %s", domain_dest)
    else:
        logger.warning("Domain file not found: %s", domain_file)

    # 5. Save config snapshot
    config = {
        "dataset": dataset_name,
        "timestamp": timestamp,
        "env": {
            k: v for k, v in os.environ.items()
            if k.startswith("DATASET_") or k.endswith("_REFRESH") or k.startswith("ERROR_")
        }
    }

    # 6. Save accuracy
    if run_accuracy:
        with open(accuracy_path, "a", encoding="utf-8") as f:
            json.dump({timestamp: run_accuracy}, f)
            f.write("\n")
        logger.debug("Saved accuracy snapshot to %s", accuracy_path)

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    logger.debug("Saved config snapshot to %s", config_path)

    logger.info("Run snapshot saved successfully.")

    return run_dir

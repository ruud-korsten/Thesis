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
    dirty_dataset_path: str,
    rules_path="artifacts/dq_rules.json",
    notes_path="artifacts/dq_notes.json",
    note_functions_path="artifacts/note_functions.json",
    output_dir="artifacts/runs"
):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    logger.info("Saving run snapshot for dataset '%s' to %s", dataset_name, run_dir)

    # 1. Save prediction mask
    pred_path = os.path.join(run_dir, "prediction_mask.csv")
    pred_mask.to_csv(pred_path, index=False)
    logger.debug("Saved prediction mask to %s", pred_path)

    # 2. Copy ground truth and dataset
    gt_dest = os.path.join(run_dir, "ground_truth_mask.xlsx")
    shutil.copy(true_mask_path, gt_dest)
    logger.debug("Copied ground truth mask to %s", gt_dest)

    dataset_dest = os.path.join(run_dir, "dirty_dataset.xlsx")
    shutil.copy(dirty_dataset_path, dataset_dest)
    logger.debug("Copied dirty dataset to %s", dataset_dest)

    # 3. Copy rule/note artifacts
    for file, name in [(rules_path, "rules.json"), (notes_path, "notes.json"), (note_functions_path, "note_functions.json")]:
        dest = os.path.join(run_dir, name)
        if os.path.exists(file):
            shutil.copy(file, dest)
            logger.debug("Copied %s to %s", file, dest)
        else:
            logger.warning("Artifact not found: %s", file)

    # 4. Save config snapshot
    config = {
        "dataset": dataset_name,
        "timestamp": timestamp,
        "env": {
            k: v for k, v in os.environ.items()
            if k.startswith("DATASET_") or k.endswith("_REFRESH") or k.startswith("ERROR_")
        }
    }

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    logger.debug("Saved config snapshot to %s", config_path)

    logger.info("✅ Run snapshot saved successfully.")
    return run_dir

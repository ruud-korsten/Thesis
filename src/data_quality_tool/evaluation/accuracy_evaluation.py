import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from data_quality_tool.config.logging_config import (
    get_logger,  # Assumes you have a logging_config module
)

logger = get_logger()

def get_ground_truth_mask_path(dataset_name: str, dirty: bool = True, mask_filename: str = None) -> str:
    """
    Dynamically constructs the ground truth mask path based on dataset name and dirty flag.
    """
    subfolder = "dirty" if dirty else "raw"
    default_filename = f"{dataset_name}_dq_mask.xlsx"
    filename = mask_filename or default_filename
    return os.path.join("data", dataset_name, subfolder, filename)

def load_mask(file_path: str) -> pd.DataFrame:
    """Loads a mask file (CSV or Excel) and normalizes its values to booleans."""
    logger.info("Loading mask from: %s", file_path)

    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path)
        else:
            logger.error("Unsupported file type for: %s", file_path)
            raise ValueError("File must be .csv, .xls, or .xlsx")
    except Exception as e:
        logger.exception("Failed to load mask file: %s", str(e))
        raise

    # Normalize values: treat non-FALSE/NaN values as violations
    normalized_df = df.astype(object)

    logger.debug("Loaded mask shape: %s, columns: %s", normalized_df.shape, list(normalized_df.columns))
    return normalized_df


def evaluate_dq_performance(true_mask: pd.DataFrame, pred_mask: pd.DataFrame) -> dict:
    if true_mask.shape != pred_mask.shape:
        logger.error("Shape mismatch detected:")
        logger.error("True mask shape: %s", true_mask.shape)
        logger.error("Pred mask shape: %s", pred_mask.shape)
        raise ValueError("Shape mismatch")
    if not all(true_mask.columns == pred_mask.columns):
        raise ValueError("Column mismatch")

    issue_types = ['missing', 'type_mismatch', 'outliers', 'duplicates']
    issue_summary = {}

    for issue in issue_types:
        injected = true_mask.applymap(lambda x: issue in str(x).split(",") if pd.notna(x) else False)
        inserted_count = int(injected.values.sum())
        detected_count = int(((injected) & pred_mask.notna()).values.sum())

        issue_summary[issue] = {
            "inserted": inserted_count,
            "detected": detected_count,
            "detection_rate": round(detected_count / inserted_count, 4) if inserted_count > 0 else 0.0,
        }

    # === Binary mask for evaluation ===
    true_binary = true_mask.applymap(lambda x: pd.notna(x) and x != "noop")
    pred_binary = pred_mask.notna()

    y_true = true_binary.values.flatten().astype(int)
    y_pred = pred_binary.values.flatten().astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    logger.debug("Total issue cells in true_mask (excluding 'noop'): %d", true_binary.values.sum())
    logger.debug("Total predicted issue cells: %d", pred_binary.values.sum())
    logger.debug("TP: %d | FP: %d | FN: %d | TN: %d", tp, fp, fn, tn)

    # === Optional: Sample false negatives ===
    missed_mask = (true_binary == 1) & (pred_binary == 0)
    missed_cells = missed_mask[missed_mask].stack()
    logger.debug("False negative (missed) cell count: %d", len(missed_cells))
    logger.debug("Sample false negatives (up to 10):\n%s", missed_cells.head(10))

    # === Optional: Save full missed set for debugging ===
    missed_df = true_mask.copy()
    for col in missed_df.columns:
        missed_df[col] = missed_df[col].where(missed_mask[col], None)
    missed_df = missed_df.dropna(how='all')
    missed_df.to_csv("artifacts/false_negatives.csv", index=False)
    logger.info("Saved missed prediction cells to artifacts/false_negatives.csv")

    # === Final metrics ===
    overall_metrics = {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    summary_df = pd.DataFrame(issue_summary).T
    logger.info("\nDQ Issue Detection Summary:\n%s", summary_df.to_string())

    metrics_df = pd.DataFrame([overall_metrics], index=["overall"]).round(4)
    logger.info("\nDQ Overall Accuracy Metrics:\n%s", metrics_df.to_string())

    return {
        "issue_detection_summary": issue_summary,
        "overall_metrics": overall_metrics
    }



def evaluate_from_files(true_mask_path: str, pred_mask_path: str) -> dict:
    """Loads masks from files (CSV or Excel) and evaluates performance metrics."""
    true_mask = load_mask(true_mask_path)
    pred_mask = load_mask(pred_mask_path)

    # Final validation before evaluation
    if true_mask.empty or pred_mask.empty:
        logger.warning("One of the masks is empty. True mask empty: %s, Pred mask empty: %s",
                       true_mask.empty, pred_mask.empty)

    return evaluate_dq_performance(true_mask, pred_mask)

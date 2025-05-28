import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_quality_tool.config.logging_config import get_logger  # Assumes you have a logging_config module

logger = get_logger()


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
    normalized_df = df.applymap(lambda x: False if pd.isna(x) or str(x).strip().upper() == "FALSE" else True)

    logger.debug("Loaded mask shape: %s, columns: %s", normalized_df.shape, list(normalized_df.columns))
    return normalized_df


def evaluate_dq_performance(true_mask: pd.DataFrame, pred_mask: pd.DataFrame) -> dict:
    """Calculates performance metrics comparing true and predicted masks, with detailed validation."""
    logger.info("Evaluating DQ performance...")

    # Shape and Column Validation
    if true_mask.shape != pred_mask.shape:
        logger.error("Shape mismatch. True mask shape: %s, Predicted mask shape: %s", true_mask.shape, pred_mask.shape)
        raise ValueError("Shape mismatch between true and predicted masks.")
    if not all(true_mask.columns == pred_mask.columns):
        mismatched_cols = [col for col in true_mask.columns if col not in pred_mask.columns]
        logger.error("Column mismatch detected. Mismatched columns: %s", mismatched_cols)
        raise ValueError("Column mismatch between true and predicted masks.")

    y_true = true_mask.values.flatten()
    y_pred = pred_mask.values.flatten()

    # Sanity Checks
    logger.debug("Total Cells Evaluated: %d", len(y_true))
    logger.debug("True Violations Count: %d", np.sum(y_true))
    logger.debug("Predicted Violations Count: %d", np.sum(y_pred))

    if np.all(y_true == False):
        logger.warning("True mask has no violations (all FALSE). Check if ground truth is correct.")

    if np.all(y_pred == False):
        logger.warning("Predicted mask has no violations (all FALSE). The model may not be detecting any issues.")

    # Compute Metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    logger.info("Evaluation Results: %s", metrics)
    return metrics


def evaluate_from_files(true_mask_path: str, pred_mask_path: str) -> dict:
    """Loads masks from files (CSV or Excel) and evaluates performance metrics."""
    true_mask = load_mask(true_mask_path)
    pred_mask = load_mask(pred_mask_path)

    # Final validation before evaluation
    if true_mask.empty or pred_mask.empty:
        logger.warning("One of the masks is empty. True mask empty: %s, Pred mask empty: %s",
                       true_mask.empty, pred_mask.empty)

    return evaluate_dq_performance(true_mask, pred_mask)

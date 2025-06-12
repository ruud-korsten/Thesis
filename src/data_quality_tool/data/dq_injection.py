import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from data_quality_tool.config.logging_config import get_logger

logger = get_logger()

load_dotenv()
INJECTION_FORCE_REFRESH = os.getenv("INJECTION_FORCE_REFRESH", "False").lower() == "true"
MAX_EXCEL_ROWS = int(os.getenv("MAX_EXCEL_ROWS", "500000"))


def get_error_config():
    return {
        'missing': float(os.getenv("ERROR_MISSING", "0.1")),
        'type_mismatch': float(os.getenv("ERROR_TYPE_MISMATCH", "0.0")),
        'outliers': float(os.getenv("ERROR_OUTLIERS", "0.0")),
        'duplicates': float(os.getenv("ERROR_DUPLICATES", "0.0")),
    }


class BaseDQInjector:
    def __init__(self, error_config=None):
        self.error_config = error_config or get_error_config()

    def _prepare_columns_for_nan(self, df: pd.DataFrame):
        changed = []
        for col in df.columns:
            if pd.api.types.is_integer_dtype(df[col]):
                df[col] = df[col].astype(float)
                changed.append((col, 'int -> float'))
            elif pd.api.types.is_bool_dtype(df[col]):
                df[col] = df[col].astype(object)
                changed.append((col, 'bool -> object'))
            elif isinstance(df[col].dtype, pd.CategoricalDtype):
                df[col] = df[col].astype(object)
                changed.append((col, 'category -> object'))
        logger.info("Columns prepared for NaN injection: %s", changed)
        return df

    def _sample_if_too_large(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) > MAX_EXCEL_ROWS:
            logger.warning("Dataset too large. Sampling down to %d rows.", MAX_EXCEL_ROWS)
            return df.sample(n=MAX_EXCEL_ROWS, random_state=42).reset_index(drop=True)
        return df

    def _should_skip_injection(self, output_dir_path: str, file_prefix: str) -> bool:
        dirty_file = Path(output_dir_path) / f"{file_prefix}.xlsx"
        mask_file = Path(output_dir_path) / f"{file_prefix}_dq_mask.xlsx"
        if not INJECTION_FORCE_REFRESH and dirty_file.exists() and mask_file.exists():
            logger.info("Skipping injection (cached dirty and mask files exist).")
            return True
        return False

    def inject_errors(self, df: pd.DataFrame, output_dir_path: str, file_prefix: str, domain: str):
        output_dir = Path(output_dir_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._should_skip_injection(output_dir_path, file_prefix):
            return
        expected_dtypes = df.apply(pd.api.types.infer_dtype)
        df = self._prepare_columns_for_nan(df)
        dq_mask = pd.DataFrame(np.nan, index=df.index, columns=df.columns, dtype=object)
        num_rows = len(df)
        num_cells = df.size

        # === OUTLIERS ===
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        num_outliers = int(self.error_config['outliers'] * num_rows)
        if num_outliers > 0 and not numeric_cols.empty:
            logger.info("Injecting %d outliers...", num_outliers)
            selected_cols = np.random.choice(numeric_cols, size=num_outliers)
            selected_rows = np.random.choice(df.index, size=num_outliers, replace=False)
            for col, row in zip(selected_cols, selected_rows):
                col_mean = df[col].mean()
                col_std = df[col].std()
                df.at[row, col] = col_mean + 10 * col_std if pd.notna(col_std) else col_mean
                dq_mask.at[row, col] = "outliers"

        # === TYPE MISMATCHES ===
        num_type_errors = int(self.error_config['type_mismatch'] * num_cells)
        if num_type_errors > 0:
            logger.info("Injecting %d type mismatches...", num_type_errors)

            numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            if not numeric_columns:
                logger.warning("No numeric columns found for type mismatch injection.")
            else:
                for col in numeric_columns:
                    df[col] = df[col].astype(object)  # ensure flexibility for string injection

                errors_injected = 0
                attempts = 0
                max_attempts = num_type_errors * 10

                while errors_injected < num_type_errors and attempts < max_attempts:
                    i = random.randint(0, num_rows - 1)
                    col_name = random.choice(numeric_columns)
                    j = df.columns.get_loc(col_name)

                    # Avoid overlap with other error types
                    if pd.isna(dq_mask.iat[i, j]):
                        df.iat[i, j] = "error_value"
                        dq_mask.iat[i, j] = "type_mismatch"
                        errors_injected += 1

                    attempts += 1

                logger.info("Successfully injected %d type mismatches.", errors_injected)

        # === MISSING VALUES ===
        num_missing = int(self.error_config['missing'] * num_cells)
        if num_missing > 0:
            logger.info("Injecting %d missing values...", num_missing)
            missing_indices = np.random.choice(num_cells, size=num_missing, replace=False)
            rows, cols = np.unravel_index(missing_indices, (num_rows, len(df.columns)))
            for r, c in zip(rows, cols):
                df.iat[r, c] = np.nan
                dq_mask.iat[r, c] = "missing"

        # === DUPLICATES ===
        num_duplicates = int(self.error_config['duplicates'] * num_rows)
        if num_duplicates > 0:
            logger.info("Injecting %d duplicate rows...", num_duplicates)
            duplicate_rows = df.sample(n=num_duplicates, replace=True, random_state=42)
            df = pd.concat([df, duplicate_rows], ignore_index=True)
            duplicate_mask = pd.DataFrame(
                [["duplicates"] * len(df.columns)] * num_duplicates,
                columns=df.columns
            )
            dq_mask = pd.concat([dq_mask, duplicate_mask], ignore_index=True)

        # === Final Validation ===
        if len(df) != len(dq_mask):
            raise ValueError(f"Mismatch before saving: df has {len(df)} rows, mask has {len(dq_mask)} rows")

        # === Prevent Excel from dropping rows with all NaNs ===
        empty_mask_rows = dq_mask.isna().all(axis=1)
        if empty_mask_rows.any():
            logger.warning("Patching %d fully empty rows in dq_mask before saving...", empty_mask_rows.sum())
            first_col = dq_mask.columns[0]
            dq_mask.loc[empty_mask_rows, first_col] = "noop"

        # === SAVE RESULTS ===
        dirty_file = output_dir / f"{file_prefix}.xlsx"
        mask_file = output_dir / f"{file_prefix}_dq_mask.xlsx"

        df.to_excel(dirty_file, index=False)
        dq_mask.to_excel(mask_file, index=False)
        print(f"Rows in df: {len(df)}")
        print(f"Rows in mask: {len(dq_mask)}")

        # Reload from saved file
        df_saved = pd.read_excel(dirty_file)
        mask_saved = pd.read_excel(mask_file)
        print(f"Rows in saved df: {len(df_saved)}")
        print(f"Rows in saved mask: {len(mask_saved)}")

        logger.info("%s DQ injection completed. Shape: %s", domain.capitalize(), df.shape)


# === Domain-specific Injectors ===

class HealthcareDQInjector(BaseDQInjector):
    def inject_errors(self, df: pd.DataFrame, output_dir_path: str, file_prefix: str = "health"):
        df = self._sample_if_too_large(df)
        logger.info("Starting Healthcare DQ Injection...")
        super().inject_errors(df, output_dir_path, file_prefix, domain="healthcare")


class RetailDQInjector(BaseDQInjector):
    def inject_errors(self, df: pd.DataFrame, output_dir_path: str, file_prefix: str = "retail"):
        df = self._sample_if_too_large(df)
        logger.info("Starting Retail DQ Injection...")
        super().inject_errors(df, output_dir_path, file_prefix, domain="retail")


class WallmartDQInjector(BaseDQInjector):
    def inject_errors(self, df: pd.DataFrame, output_dir_path: str, file_prefix: str = "wallmart"):
        df = self._sample_if_too_large(df)
        logger.info("Starting Wallmart DQ Injection...")
        super().inject_errors(df, output_dir_path, file_prefix, domain="wallmart")

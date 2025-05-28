import pandas as pd
import numpy as np
import random
from pathlib import Path
from data_quality_tool.config.logging_config import get_logger  # Proper logger import

logger = get_logger()

MAX_EXCEL_ROWS = 500_000  # Excel export safeguard


class BaseDQInjector:
    def __init__(self, error_config=None):
        self.error_config = error_config or {
            'missing': 0.1,
            'type_mismatch': 0.005,
            'outliers': 0.01,
            'duplicates': 0.01,
            'format_inconsistency': 0.005
        }

    def inject_errors(self, df: pd.DataFrame, output_dir_path: str, file_prefix: str):
        raise NotImplementedError("This method should be implemented by subclasses.")


class HealthcareDQInjector(BaseDQInjector):
    def inject_errors(self, df: pd.DataFrame, output_dir_path: str, file_prefix: str = "health"):
        logger.info("Starting Healthcare DQ Injection...")
        self._inject_generic_errors(df, output_dir_path, file_prefix, domain="healthcare")


class RetailDQInjector(BaseDQInjector):
    def inject_errors(self, df: pd.DataFrame, output_dir_path: str, file_prefix: str = "retail"):
        logger.info("Starting Retail DQ Injection...")
        self._inject_generic_errors(df, output_dir_path, file_prefix, domain="retail")


    def _inject_generic_errors(self, df: pd.DataFrame, output_dir_path: str, file_prefix: str, domain: str):
        dq_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        num_cells = df.size
        num_rows, num_cols = df.shape

        # === Missing Values ===
        num_missing = int(self.error_config['missing'] * num_cells)
        if num_missing > 0:
            logger.info("Injecting %d missing values...", num_missing)
            missing_indices = np.random.choice(num_cells, size=num_missing, replace=False)
            rows, cols = np.unravel_index(missing_indices, (num_rows, num_cols))
            df.values[rows, cols] = np.nan
            dq_mask.values[rows, cols] = True

        # === Type Mismatches ===
        num_type_errors = int(self.error_config['type_mismatch'] * num_cells)
        if num_type_errors > 0:
            logger.info("Injecting %d type mismatches...", num_type_errors)
            for _ in range(num_type_errors):
                i, j = random.randint(0, num_rows - 1), random.randint(0, num_cols - 1)
                col_name = df.columns[j]
                if pd.api.types.is_numeric_dtype(df[col_name]):
                    df.iat[i, j] = "error_value"
                    dq_mask.iat[i, j] = True

        # === Outliers Injection ===
        if domain == "healthcare" and 'Age' in df.columns:
            df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
            num_outliers = int(self.error_config['outliers'] * num_rows)
            logger.info("Injecting %d age outliers...", num_outliers)
            outlier_indices = np.random.choice(df.index, size=num_outliers, replace=False)
            df.loc[outlier_indices, 'Age'] = 999
            dq_mask.loc[outlier_indices, 'Age'] = True

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        num_outliers = int(self.error_config['outliers'] * num_rows * len(numeric_cols))
        if num_outliers > 0 and not numeric_cols.empty:
            logger.info("Injecting %d numeric outliers...", num_outliers)
            col_choices = np.random.choice(numeric_cols, size=num_outliers)
            row_choices = np.random.randint(0, num_rows, size=num_outliers)
            for col, row in zip(col_choices, row_choices):
                if not pd.api.types.is_float_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                col_mean = df[col].mean()
                col_std = df[col].std()
                outlier_value = col_mean + 10 * col_std if pd.notna(col_std) else col_mean
                df.at[row, col] = outlier_value
                dq_mask.at[row, col] = True

        # === Duplicates Injection ===
        num_duplicates = int(self.error_config['duplicates'] * num_rows)
        if num_duplicates > 0:
            logger.info("Injecting %d duplicate rows...", num_duplicates)
            duplicate_rows = df.sample(n=num_duplicates, replace=True)
            df_with_duplicates = pd.concat([df, duplicate_rows], ignore_index=True)
            dq_mask_with_duplicates = pd.concat(
                [dq_mask, pd.DataFrame(True, index=range(len(df), len(df) + num_duplicates), columns=df.columns)],
                ignore_index=True
            )
        else:
            df_with_duplicates = df
            dq_mask_with_duplicates = dq_mask

        # === Format Inconsistencies ===
        if domain == "healthcare" and 'Gender' in df.columns:
            gender_map = {'Male': 'male', 'Female': 'FEMALE'}
            df['Gender'] = df['Gender'].replace(gender_map)
            dq_mask['Gender'] = dq_mask['Gender'] | df['Gender'].isin(gender_map.values())

        if 'Lab_Results' in df.columns:
            num_format_errors = int(self.error_config['format_inconsistency'] * num_rows)
            logger.info("Injecting %d lab result format errors...", num_format_errors)
            format_error_indices = np.random.choice(df.index, size=num_format_errors, replace=False)
            df.loc[format_error_indices, 'Lab_Results'] = "Invalid Result"
            dq_mask.loc[format_error_indices, 'Lab_Results'] = True

        # === Sampling for Excel Export ===
        if len(df_with_duplicates) > MAX_EXCEL_ROWS:
            logger.warning("Dataset too large. Sampling down to %d rows.", MAX_EXCEL_ROWS)
            sample_indices = np.random.choice(df_with_duplicates.index, size=MAX_EXCEL_ROWS, replace=False)
            df_to_save = df_with_duplicates.loc[sample_indices]
            mask_to_save = dq_mask_with_duplicates.loc[sample_indices]
        else:
            df_to_save = df_with_duplicates
            mask_to_save = dq_mask_with_duplicates

        # === Save Results ===
        output_dir = Path(output_dir_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        dirty_file = output_dir / f"{file_prefix}_dirty.xlsx"
        mask_file = output_dir / f"{file_prefix}_dq_mask.xlsx"

        logger.info("Saving dirty dataset to: %s", dirty_file)
        logger.info("Saving DQ mask to: %s", mask_file)

        df_to_save.to_excel(dirty_file, index=False)
        mask_to_save.to_excel(mask_file, index=False)

        logger.info("%s DQ injection completed successfully.", domain.capitalize())

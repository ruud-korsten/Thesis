import pandas as pd
from data_quality_tool.config.logging_config import get_logger

logger = get_logger()


class BaseDQCheck:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.reports = {}
        logger.debug("Initialized BaseDQCheck with dataframe of shape %s", self.df.shape)

    def _add_report(self, name: str, report_df: pd.DataFrame):
        self.reports[name] = report_df
        logger.info("Report '%s' added with shape %s", name, report_df.shape)

    def get_reports(self):
        return self.reports


class MissingValueCheck(BaseDQCheck):
    def check(self):
        logger.info("Running missing value check")
        missing = self.df.isnull().sum()
        percent = (missing / len(self.df)) * 100

        mask = missing > 0
        report = pd.DataFrame({
            'missing_values': missing[mask],
            'percent_missing': percent[mask]
        }).sort_values(by='percent_missing', ascending=False)

        self._add_report('missing', report)
        logger.debug("Missing value report generated with %d columns having missing values", mask.sum())
        return report


class DuplicateCheck(BaseDQCheck):
    def check(self, subset=None, keep='first'):
        logger.info("Running duplicate check (subset=%s, keep=%s)", subset, keep)
        duplicates = self.df.duplicated(subset=subset, keep=keep)
        duplicate_rows = self.df[duplicates].copy()
        report = pd.DataFrame({
            'total_rows': [len(self.df)],
            'duplicate_rows': [duplicates.sum()],
            'percent_duplicates': [round(duplicates.sum() / len(self.df) * 100, 4)]
        })

        self._add_report('duplicates', report)
        self._add_report('duplicate_rows', duplicate_rows)
        logger.debug("Found %d duplicate rows", duplicates.sum())
        return duplicate_rows


class DataQualityChecker:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.reports = {}
        logger.info("DataQualityChecker initialized with dataframe of shape %s", df.shape)

    def run_all_checks(self):
        logger.info("Running all data quality checks")

        mv = MissingValueCheck(self.df)
        mv.check()
        self.reports.update(mv.get_reports())

        dc = DuplicateCheck(self.df)
        dc.check()
        self.reports.update(dc.get_reports())

        logger.info("All checks completed. Reports generated: %s", list(self.reports.keys()))
        return self.reports

    def generate_violation_mask(self) -> pd.DataFrame:
        """Generates a cell-level mask for standard DQ issues."""
        logger.info("Generating standard DQ violation mask...")
        mask_df = pd.DataFrame("", index=self.df.index, columns=self.df.columns)

        # Missing Values
        missing_mask = self.df.isnull()
        mask_df = mask_df.mask(missing_mask, "MISSING")

        # Duplicates (flag all columns in duplicate rows)
        duplicates = self.df.duplicated(keep=False)
        for idx in self.df[duplicates].index:
            for col in self.df.columns:
                current_flag = mask_df.at[idx, col]
                mask_df.at[idx, col] = f"{current_flag}, DUPLICATE" if current_flag else "DUPLICATE"

        logger.info("Standard DQ violation mask generation complete.")
        return mask_df

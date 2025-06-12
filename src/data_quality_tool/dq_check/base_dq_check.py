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
        print(report)
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


class MismatchCheck(BaseDQCheck):
    def __init__(self, df, expected_dtypes=None):
        super().__init__(df)
        self.expected_dtypes = expected_dtypes

    def is_mismatch(self, val, expected_type):
        if pd.isna(val):
            return False
        try:
            if expected_type.kind in 'iufc':  # numeric types
                float(val)
            elif expected_type.kind == 'b':  # boolean
                return not isinstance(val, bool)
            elif expected_type.kind == 'M':  # datetime
                pd.to_datetime(val)
            elif expected_type.kind in 'OUS':  # string-like
                str(val)
            else:
                return False
            return False
        except Exception:
            return True

    def check(self):
        logger.info("Running schema mismatch check using expected_dtypes")
        mismatch_counts = {}
        percent_mismatches = {}

        for col in self.expected_dtypes.index:
            if col not in self.df.columns:
                continue
            expected_type = self.expected_dtypes[col]
            mismatches = self.df[col].apply(lambda x: self.is_mismatch(x, expected_type))
            mismatch_counts[col] = mismatches.sum()
            percent_mismatches[col] = round((mismatches.sum() / len(self.df)) * 100, 2)

        mismatch_series = pd.Series(mismatch_counts)
        percent_series = pd.Series(percent_mismatches)
        mask = mismatch_series > 0

        report = pd.DataFrame({
            'mismatched_values': mismatch_series[mask],
            'percent_mismatched': percent_series[mask],
            'expected_dtype': self.expected_dtypes[mask].astype(str)
        }).sort_values(by='percent_mismatched', ascending=False)

        self._add_report('schema_mismatches', report)
        logger.debug("Schema mismatch report generated with %d columns having mismatches", mask.sum())
        return report




class DataQualityChecker:
    def __init__(self, df: pd.DataFrame, expected_dtypes=None):
        self.df = df
        self.expected_dtypes = expected_dtypes
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

        logger.info(self.expected_dtypes)
        mm = MismatchCheck(self.df, expected_dtypes=self.expected_dtypes)
        mm.check()
        self.reports.update(mm.get_reports())

        logger.info("All checks completed. Reports generated: %s", list(self.reports.keys()))
        return self.reports

    def generate_violation_mask(self) -> pd.DataFrame:
        """Generates a cell-level mask for standard DQ issues."""
        logger.info("Generating standard DQ violation mask...")
        mask_df = pd.DataFrame("", index=self.df.index, columns=self.df.columns)

        # === Missing Values
        missing_mask = self.df.isnull()
        mask_df = mask_df.mask(missing_mask, "MISSING")

        # === Duplicates
        duplicates = self.df.duplicated(keep="first")
        for idx in self.df[duplicates].index:
            for col in self.df.columns:
                current_flag = mask_df.at[idx, col]
                mask_df.at[idx, col] = f"{current_flag}, DUPLICATE" if current_flag else "DUPLICATE"

        # === Type Mismatches
        if hasattr(self, "expected_dtypes") and self.expected_dtypes is not None:
            logger.info("Adding type mismatches to DQ mask...")
            mm_check = MismatchCheck(self.df, expected_dtypes=self.expected_dtypes)

            for col in self.df.columns:
                if col not in self.expected_dtypes:
                    continue
                expected_type = self.expected_dtypes[col]
                mismatches = self.df[col].apply(lambda x: mm_check.is_mismatch(x, expected_type))
                idx = mismatches[mismatches].index

                for i in idx:
                    current_flag = mask_df.at[i, col]
                    mask_df.at[i, col] = f"{current_flag}, TYPE_MISMATCH" if current_flag else "TYPE_MISMATCH"

        logger.info("Standard DQ violation mask generation complete.")
        return mask_df


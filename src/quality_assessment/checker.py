import pandas as pd

class BaseDQCheck:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.reports = {}

    def _add_report(self, name: str, report_df: pd.DataFrame):
        self.reports[name] = report_df

    def get_reports(self):
        return self.reports


class MissingValueCheck(BaseDQCheck):
    def check(self):
        missing = self.df.isnull().sum()
        percent = (missing / len(self.df)) * 100

        # Filter only columns where missing values > 0
        mask = missing > 0
        report = pd.DataFrame({
            'missing_values': missing[mask],
            'percent_missing': percent[mask]
        }).sort_values(by='percent_missing', ascending=False)

        self._add_report('missing', report)
        return report


class DuplicateCheck(BaseDQCheck):
    def check(self, subset=None, keep='first'):
        duplicates = self.df.duplicated(subset=subset, keep=keep)
        duplicate_rows = self.df[duplicates].copy()
        report = pd.DataFrame({
            'total_rows': [len(self.df)],
            'duplicate_rows': [duplicates.sum()],
            'percent_duplicates': [round(duplicates.sum() / len(self.df) * 100, 4)]
        })

        self._add_report('duplicates', report)
        self._add_report('duplicate_rows', duplicate_rows)
        return duplicate_rows


class DataQualityChecker:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.reports = {}

    def run_all_checks(self):
        mv = MissingValueCheck(self.df)
        mv.check()
        self.reports.update(mv.get_reports())

        dc = DuplicateCheck(self.df)
        dc.check()
        self.reports.update(dc.get_reports())

        return self.reports
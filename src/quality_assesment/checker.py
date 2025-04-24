import pandas as pd

class DataQualityChecker:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.reports = {}

    def check_missing_values(self):
        missing = self.df.isnull().sum()
        percent = (missing / len(self.df)) * 100
        report = pd.DataFrame({
            'missing_values': missing,
            'percent_missing': percent
        }).sort_values(by='percent_missing', ascending=False)
        self.reports['missing'] = report
        return report

    def detect_outliers_iqr_grouped(self, group_cols=None, target_cols=None, multiplier=1.5, verbose=True):
        df_copy = self.df.copy()

        if target_cols is None:
            target_cols = df_copy.select_dtypes(include='number').columns.tolist()

        if group_cols is None:
            groupby_obj = [(None, df_copy)]
        else:
            groupby_obj = df_copy.groupby(group_cols)

        summary_reports = {}

        for target in target_cols:
            is_outlier_series = pd.Series(False, index=df_copy.index)

            for _, group_df in groupby_obj:
                if group_df[target].nunique() < 2:
                    continue  # Skip groups with no variance

                Q1 = group_df[target].quantile(0.1)
                Q3 = group_df[target].quantile(0.9)
                IQR = Q3 - Q1
                lower = Q1 - multiplier * IQR
                upper = Q3 + multiplier * IQR

                mask = (group_df[target] < lower) | (group_df[target] > upper)
                is_outlier_series.loc[mask.index[mask]] = True

            col_flag_name = f'is_outlier_{target}'
            df_copy[col_flag_name] = is_outlier_series

            if verbose:
                count = is_outlier_series.sum()
                print(f"[{col_flag_name}] Outliers detected: {count}")

            # --- Create summary report for this target column ---
            if group_cols is not None:
                summary_df = df_copy.groupby(group_cols).agg(
                    outlier_count=(col_flag_name, 'sum'),
                    total_count=(target, 'count')
                ).reset_index()
                summary_df['percent_outliers'] = (summary_df['outlier_count'] / summary_df['total_count']) * 100
                summary_df = summary_df.sort_values(by='percent_outliers', ascending=False)
                summary_reports[f'outliers_summary_{target}'] = summary_df

        self.reports['outliers_grouped'] = df_copy[[col for col in df_copy.columns if col.startswith('is_outlier_')]]

        # Store summaries in the reports dictionary
        self.reports.update(summary_reports)

        return df_copy

    def check_duplicate_rows(self, subset: list[str] = None, keep='first') -> pd.DataFrame:
        """
        Checks for duplicate rows in the dataset.

        Args:
            subset: List of columns to consider for duplicates. If None, checks all columns.
            keep: Which duplicates to mark: 'first', 'last', or False (mark all duplicates).

        Returns:
            DataFrame with duplicate rows.
        """
        duplicates = self.df.duplicated(subset=subset, keep=keep)
        duplicate_rows = self.df[duplicates].copy()

        # Summary report
        report = pd.DataFrame({
            'total_rows': [len(self.df)],
            'duplicate_rows': [duplicates.sum()],
            'percent_duplicates': [round(duplicates.sum() / len(self.df) * 100, 4)]
        })

        self.reports['duplicates'] = report
        self.reports['duplicate_rows'] = duplicate_rows

        return duplicate_rows


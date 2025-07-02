import re
import pandas as pd

from data_quality_tool.config.logging_config import get_logger

logger = get_logger()

class NoteEvaluator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.passed = {}
        self.failed = {}
        self.df_columns = set(df.columns)
        logger.debug("NoteEvaluator initialized with DataFrame of shape %s", df.shape)

    def _extract_columns_from_code(self, code: str) -> list:
        """
        Extracts DataFrame column names used in the code using regex.
        """
        pattern = r"df\[['\"]([a-zA-Z0-9_ ]+)['\"]\]"
        matches = re.findall(pattern, code)
        used_cols = list({col for col in matches if col in self.df_columns})
        return used_cols

    def evaluate(self, note_functions: dict[str, str]) -> dict[str, dict]:
        self.passed = {}
        self.failed = {}

        for idx, (note, code) in enumerate(note_functions.items()):
            note_id = f"N{idx + 1:03d}"
            logger.info("Evaluating note %s: %s", note_id, note)
            local_env = {}

            try:
                exec(code, {"pd": pd}, local_env)
                func = next((v for v in local_env.values() if callable(v)), None)
                if func is None:
                    raise ValueError("No valid function was defined in the code block.")

                result = func(self.df)

                if not isinstance(result, pd.Series):
                    raise ValueError("Function did not return a pandas Series.")
                if result.dtype != bool:
                    raise ValueError("Returned Series is not boolean.")
                if len(result) != len(self.df):
                    raise ValueError("Returned Series length does not match DataFrame.")
                if not result.index.equals(self.df.index):
                    raise ValueError("Returned Series index does not match DataFrame index.")

                num_flags = result.sum()

                # === Enhanced column extraction ===
                columns_used = self._extract_columns_from_code(code)
                if not columns_used and result.name in self.df.columns:
                    columns_used = [result.name]

                self.passed[note] = {
                    "id": note_id,
                    "violations": int(num_flags),
                    "function_name": func.__name__,
                    "violations_mask": result,
                    "code": code,
                    "columns": columns_used,
                }

                logger.info("%s flagged %d rows on columns: %s", func.__name__, num_flags, columns_used)

            except Exception as e:
                error_msg = str(e)
                logger.warning("Skipping note check %s due to error: %s", note_id, error_msg)

                self.failed[note] = {
                    "id": note_id,
                    "code": code,
                    "error": error_msg,
                }

        return {
            "passed": self.passed,
            "failed": self.failed,
        }

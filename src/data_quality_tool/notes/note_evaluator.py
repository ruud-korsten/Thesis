import pandas as pd
from data_quality_tool.logging_config import get_logger

logger = get_logger()


class NoteEvaluator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}
        logger.debug("NoteEvaluator initialized with DataFrame of shape %s", df.shape)

    def evaluate(self, note_functions: dict[str, str]) -> dict[str, dict]:
        self.results = {}

        for idx, (note, code) in enumerate(note_functions.items()):
            note_id = f"N{idx + 1:03d}"
            logger.info("Evaluating note %s: %s", note_id, note)
            local_env = {}

            try:
                # Ensure pandas is available
                exec(code, {"pd": pd}, local_env)

                # Find the first callable in the local env
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

                num_flags = result.sum()

                self.results[note] = {
                    "id": note_id,
                    "violations": int(num_flags),
                    "function_name": func.__name__,
                    "violations_mask": result,
                    "code": code
                }

                logger.info("%s flagged %d rows", func.__name__, num_flags)

            except Exception as e:
                logger.warning("Skipping note check %s due to error: %s", note_id, str(e))
                self.results[note] = {
                    "id": note_id,
                    "error": str(e),
                    "code": code
                }

        return self.results

import pandas as pd
from typing import Dict

class NoteEvaluator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}

    def evaluate(self, note_functions: Dict[str, str]) -> Dict[str, Dict]:
        self.results = {}

        for note, code in note_functions.items():
            print(f"\nEvaluating note for: {note}")
            local_env = {}

            try:
                # Execute the code string to define the function
                exec(code, {}, local_env)
                func = next(v for v in local_env.values() if callable(v))

                result = func(self.df)

                if not isinstance(result, pd.Series):
                    raise ValueError("Function did not return a pandas Series.")

                if len(result) != len(self.df):
                    raise ValueError("Returned Series length does not match DataFrame.")

                num_flags = result.sum()
                self.results[note] = {
                    "violations": int(num_flags),
                    "function_name": func.__name__,
                    "code": code
                }

                print(f"✓ {func.__name__} flagged {num_flags} rows.")

            except Exception as e:
                print(f"Error running note check: {e}")
                self.results[note] = {
                    "error": str(e),
                    "code": code
                }

        return self.results

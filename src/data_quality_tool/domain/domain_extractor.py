import os
import pandas as pd
from data_quality_tool.llm.llm_client import LLMClient
from data_quality_tool.config.logging_config import get_logger

logger = get_logger()


class DomainExtractor:
    def __init__(self, model: str = None, temperature: float = 0.6):
        self.llm = LLMClient(model=model, temperature=temperature)
        logger.info("DomainExtractor initialized successfully")

    def build_domain_messages(self, df_sample: pd.DataFrame) -> list[dict]:
        logger.debug("Building domain extraction messages using first 10 rows and schema...")

        sample_data = df_sample.head(10).to_markdown(index=False)
        schema_info = "\n".join(f"- {col}: {dtype}" for col, dtype in df_sample.dtypes.items())
        summary_stats = self._generate_summary_stats(df_sample)

        system_message = {
            "role": "system",
            "content": (
                "You are a domain expert assistant. Your task is to analyze structured datasets and infer both real-world meaning and "
                "context-specific data quality expectations for each column. Focus on expectations that help identify hidden data issues — "
                "such as out-of-range values, invalid IDs, or implausible business values — which cannot be detected from schema alone. "
                "Avoid assumptions and skip ambiguous or unclear columns. Be concise, precise, and only derive rules that are clearly supported by the data."
                "Carefully choose the appropriate quantitative indication for each column, considering whether absolute thresholds or statistical deviations are more suitable."
                "Your domain knowledge will be used to generate data validation rules and detect hidden data quality issues. Make sure they are accurate and won't flag every row in the dataset."
                "Don't give too concrete examples, as they can be misleading. "
                "Never use emoticons or use Unicode symbols; always use plain text equivalents."
            )
        }

        user_message = {
            "role": "user",
            "content": f"""
        Analyze the data sample and schema below. For each column where the meaning is clear, infer domain knowledge and **data quality expectations** that can help identify subtle or context-specific issues.

        ---

        ### **Data Sample:**
        {sample_data}

        ### **Schema:**
        {schema_info}

        ### **Summary Statistics:**
        {summary_stats}

        ---

        ### **Output Format**

        1. **Inferred Domain Knowledge**  
        - For each relevant column:
          - Provide **a clear sentence** explaining its meaning and use.
          - Specify its data type and categorize as one of: **unique ID**, **categorical**, **numerical**, **timestamp**.
          - List any **data quality expectations**, such as:
            - Valid ranges or thresholds (e.g., cost must be ≥ 0)
            - Specific format patterns (e.g., IDs like `CUST-XXXX`)
            - Flags for known problematic values (e.g., `9999`, empty strings, negative counts)
            - Extreme outliers in numerical data (e.g., more than 5 standard deviations)

        2. **Skip columns where usage cannot be confidently inferred.**

        **Example:**  
        - `usage`: Numerical; daily energy consumption in kWh. Must be => 0. Extreme values > 1000 kWh/day should be flagged.  
        - `customer_id`: Unique ID for each customer. Must follow format `CUST-XXXX`.  
        - `age`: Integer. Represents age in years; should be a positive number less than 120.  
        - `sensor_reading`: Numerical. Represents equipment vibration. Values above 300 may indicate malfunction.
        
        There can be multiple rules for each column.
        Don't consider missing values and duplicate rows, as they can be detected without context.
        Be precise, avoid speculation, and output only columns where the context clearly supports an interpretation. This information will be used to generate data validation rules and detect hidden data quality issues.
        """.strip()
        }

        logger.debug("Domain extraction messages successfully built.")
        return [system_message, user_message]

    def _generate_summary_stats(self, df: pd.DataFrame) -> str:
        """Generates classic summary statistics for numerical and categorical columns."""
        try:
            logger.debug("Generating summary statistics...")
            numeric_summary = df.describe().transpose().head(10)
            categorical_summary = df.describe(include=['object', 'category']).transpose().head(10)

            stats_sections = []
            if not numeric_summary.empty:
                stats_sections.append("#### Numerical Columns Summary:\n" + numeric_summary.to_markdown())
            if not categorical_summary.empty:
                stats_sections.append("#### Categorical Columns Summary:\n" + categorical_summary.to_markdown())

            logger.debug("Summary statistics generated successfully.")
            return "\n\n".join(stats_sections) if stats_sections else "No summary statistics available."

        except Exception as e:
            logger.warning("Failed to generate summary statistics: %s", str(e))
            return "Summary statistics unavailable."

    def extract_domain_knowledge(self, df: pd.DataFrame, save_path: str = None) -> dict:
        if df.empty:
            logger.warning("Input DataFrame is empty. Skipping domain extraction.")
            return {}

        logger.info("Starting domain knowledge extraction...")
        messages = self.build_domain_messages(df)

        try:
            response = self.llm.call(messages=messages)
            logger.info("Domain knowledge extracted successfully. Response length: %d", len(response))
        except Exception as e:
            logger.error("LLM call failed during domain extraction: %s", str(e))
            return {"prompt": messages, "response": "ERROR: LLM call failed."}

        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(response)
                logger.info("Domain knowledge saved to: %s", save_path)
            except Exception as e:
                logger.error("Failed to save domain knowledge to %s: %s", save_path, str(e))

        return {"prompt": messages, "response": response}

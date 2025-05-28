import pandas as pd
import os
from data_quality_tool.llm.llm_client import LLMClient
import logging

logger = logging.getLogger(__name__)

class DomainExtractor:
    def __init__(self, model: str = None, temperature: float = 0.2):
        self.llm = LLMClient(model=model, temperature=temperature)

    def build_prompt(self, df_sample: pd.DataFrame) -> str:
        sample_data = df_sample.head(10).to_markdown(index=False)
        schema_info = "\n".join(f"- {col}: {dtype}" for col, dtype in df_sample.dtypes.items())

        # Add descriptive statistics to the prompt
        summary_stats = self._generate_summary_stats(df_sample)

        return f"""
        You are a domain expert. Analyze the data sample and schema below. Produce **concise domain knowledge only for columns where meaning and usage are clear**.

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
        - For relevant columns, provide **one clear sentence** explaining their real-world meaning and typical use.  
        - Specify data type and categorize as **unique ID**, **categorical**, **numerical**, or **timestamp**.  
        - Where applicable, include **data quality expectations** (e.g., mandatory field, specific format, valid range).  
        - **Skip columns where the purpose cannot be reliably inferred.**

        **Example:**  
        - `usage`: Numerical; daily energy consumption in kWh. Must be a non-negative value.  
        - `customer_id`: Unique ID for each customer. Must be filled, unique, and follow the format `CUST-XXXX`.  
        - `age`: An integer with the age of someone. Must be a positive integer, within a reasonable range.  
        - 'costs': A float indicating the cost of something. Extreme outliers should be flagged.
        
        Only extract knowledge intrinsic to the columns, we don't care about ambiguous things like missing values.
        Be precise, avoid assumptions, and skip irrelevant or ambiguous fields. If no domain knowledge is confidently inferred, exclude the column entirely from the output.
        """.strip()

    def _generate_summary_stats(self, df: pd.DataFrame) -> str:
        """Generates classic summary statistics for numerical and categorical columns."""
        try:
            # Numeric Summary
            numeric_summary = df.describe().transpose()
            # Categorical Summary
            categorical_summary = df.describe(include=['object', 'category']).transpose()

            # Limit output to first 10 columns for brevity
            numeric_summary = numeric_summary.head(10)
            categorical_summary = categorical_summary.head(10)

            stats_sections = []

            if not numeric_summary.empty:
                stats_sections.append("#### Numerical Columns Summary:\n" + numeric_summary.to_markdown())

            if not categorical_summary.empty:
                stats_sections.append("#### Categorical Columns Summary:\n" + categorical_summary.to_markdown())

            return "\n\n".join(stats_sections) if stats_sections else "No summary statistics available."
        except Exception as e:
            logger.warning("Failed to generate summary statistics: %s", str(e))
            return "Summary statistics unavailable."

    def extract_domain_knowledge(self, df: pd.DataFrame, save_path: str = None) -> dict:
        if df.empty:
            logger.warning("Input DataFrame is empty. Skipping domain extraction.")
            return {}

        prompt = self.build_prompt(df)
        print(prompt)  # For inspection, can be removed in production
        response = self.llm.call(prompt)

        logger.info("Domain knowledge extracted successfully.")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(response)
            logger.info("Domain knowledge saved to %s", save_path)

        return {"prompt": prompt, "response": response}

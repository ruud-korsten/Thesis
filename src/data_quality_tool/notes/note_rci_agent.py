
from data_quality_tool.config.logging_config import get_logger
from data_quality_tool.llm.llm_client import LLMClient

logger = get_logger()


class NoteRCIAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        logger.debug("NoteRCIAgent initialized with LLM client: %s", type(llm_client).__name__)

    def build_critique_prompt(self, note: str, function_code: str, df_columns: list[str]) -> str:
        return (
            "You are a senior Python code reviewer and data quality engineer.\n\n"
            "Please critique the following function generated for a data quality note.\n\n"
            "---\n\n"
            f"Note:\n\"{note}\"\n\n"
            "Available Columns:\n"
            f"{', '.join(df_columns)}\n\n"
            "Function:\n"
            f"{function_code}\n\n"
            "---\n\n"
            "### Critique Instructions:\n"
            "- Does the function correctly express the logic described in the note?\n"
            "- Are there potential bugs, unnecessary complexity, or logic gaps?\n"
            "- Is the function Pythonic and efficient?\n"
            "- Are the assumptions about column names valid?\n"
            "- Does it return a valid boolean Series of violations?\n\n"
            "Respond with a clear, constructive critique and suggest specific improvements if needed."
            "Do **not** use EMOJIS and Unicode characters (-> in stead of arrows)!"
        )

    def build_improvement_prompt(self, note: str, original_code: str, critique: str, df_columns: list[str]) -> str:
        return (
            "You are a data validation assistant.\n\n"
            "Improve the following Python function based on the critique below.\n\n"
            "---\n\n"
            f"Note:\n\"{note}\"\n\n"
            f"Available Columns:\n{', '.join(df_columns)}\n\n"
            "Critique:\n"
            f"{critique}\n\n"
            "Original Code:\n"
            f"{original_code}\n\n"
            "---\n\n"
            "Return only the improved Python function. Do not include explanations or markdown formatting."
            "Don't import pandas. Just return the function."
            "Do **not** use EMOJIS and Unicode characters (-> in stead of arrows)!"
        )

    def run_note_rci(self, note: str, raw_code: str, df_columns: list[str]) -> dict:
        logger.info("Running RCI for note: %s", note)
        logger.debug("Initial raw code:\n%s", raw_code)

        # === Critique Step ===
        critique_prompt = self.build_critique_prompt(note, raw_code, df_columns)
        critique, critique_usage = self.llm.call(prompt=critique_prompt)
        logger.debug("Critique:\n%s", critique)

        # === Improvement Step ===
        improvement_prompt = self.build_improvement_prompt(note, raw_code, critique, df_columns)
        improved_code, improvement_usage = self.llm.call(prompt=improvement_prompt)
        logger.debug("Improved Code:\n%s", improved_code)

        # === Return Results and Flat Usage Stats ===
        return {
            "original": raw_code,
            "critique": critique,
            "improved": improved_code,
            "critique_usage": critique_usage,
            "improvement_usage": improvement_usage
        }


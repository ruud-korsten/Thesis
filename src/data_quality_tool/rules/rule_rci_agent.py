from data_quality_tool.config.logging_config import get_logger
import textwrap
logger = get_logger()


class RuleRCIAgent:
    def __init__(self, llm_client):
        self.llm = llm_client
        logger.debug("RuleRCIAgent initialized with LLM client: %s", type(llm_client).__name__)

    def build_critique_prompt(self, draft: str, domain_text: str, df_columns: list[str]) -> str:
        """
        ‘C’ prompt – audits the draft against every requirement.
        """
        columns_str = ", ".join(df_columns)
        checklist = textwrap.dedent("""
        - **Schema & formatting**
          • `Structured Rules` is a valid JSON array, no trailing commas, unique `"id"`.
          • Only two top-level sections appear.

        - **Column & rule validity**
          • Only columns listed in **Dataset Columns** are used.
          • `type` is one of: `range`, `not_null`, `conditional`, `pattern`.
          • `condition` object contains only keys that make sense.
          • `group_by` appears only when needed.

        - **Coverage & clarity**
          • No enforceable rule from **Domain Knowledge** is missing.
          • No duplicates or contradictions.
          • `name` and `message` are short and plain.

        - **Other Insights**
          • Contains only items that do **not** map to the four rule types.
        """).strip()

        return textwrap.dedent(f"""
        You are now in the **CRITIQUE** phase.  
        Check the draft extraction against every item below.

        ### Dataset Columns
        [{columns_str}]

        ### Domain Knowledge
        \"\"\"{domain_text}\"\"\"

        ### Draft Extraction
        {draft}

        ### Critique (write a bulleted list starting with "**Critique:**")
        {checklist}
        "Do **not** use EMOJIS and Unicode characters!"
        """).strip()

    def build_improvement_prompt(self,
                                 original: str,
                                 critique: str,
                                 domain_text: str,
                                 df_columns: list[str]) -> str:
        """
        ‘I’ prompt – rewrites from scratch, fixing every issue.
        """
        columns_str = ", ".join(df_columns)
        return textwrap.dedent(f"""
        You are now in the **IMPROVE** phase.

        ### Dataset Columns
        [{columns_str}]

        ### Domain Knowledge
        \"\"\"{domain_text}\"\"\"

        ### Critique
        {critique}

        ### Original Draft
        {original}

        Rewrite the answer **from scratch**, ensuring it:

        1. Fully complies with the required format.
        2. Includes all valid rules and insights, with no extras.
        3. Is self-consistent, well-named, concise.

        Output **only** the two sections, in this order:

        ### Structured Rules
        [ ... valid JSON array ... ]

        ### Other Insights
        - ...
        - ...

        Do **not** add commentary before or after those sections.
        "Do **not** use EMOJIS and Unicode characters!"
        """).strip()

    def run_rci_pipeline(self, messages: str, domain_text: str, df_columns: list[str]) -> dict:
        logger.info("Running RuleRCI pipeline")

        # Step 1: Initial call
        initial_output, usage_initial = self.llm.call(messages=messages)
        logger.debug("Initial output received:\n%s", initial_output)

        # Step 2: Critique phase
        critique_prompt = self.build_critique_prompt(initial_output, domain_text, df_columns)
        critique, usage_critique = self.llm.call(prompt=critique_prompt)
        logger.debug("Critique received:\n%s", critique)

        # Step 3: Improve phase
        improve_prompt = self.build_improvement_prompt(initial_output, critique, domain_text, df_columns)
        improved_output, usage_improve = self.llm.call(prompt=improve_prompt)
        logger.debug("Improved output received:\n%s", improved_output)

        return {
            "initial_output": initial_output,
            "critique": critique,
            "improved_output": improved_output,
            "token_usage": {
                "initial": usage_initial,
                "critique": usage_critique,
                "improve": usage_improve
            }
        }


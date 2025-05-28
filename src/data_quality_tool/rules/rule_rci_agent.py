from data_quality_tool.logging_config import get_logger

logger = get_logger()


class RuleRCIAgent:
    def __init__(self, llm_client):
        self.llm = llm_client
        logger.debug("RuleRCIAgent initialized with LLM client: %s", type(llm_client).__name__)

    def build_critique_prompt(self, output: str, domain_text: str, df_columns: list[str]) -> str:
        return f"""
You are a senior data quality engineer. Your task is to critique a rule extraction output based on the provided domain knowledge, dataset schema, and the known capabilities of the rule engine.
...
### Dataset Columns

[{", ".join(df_columns)}]

...
{domain_text}
...
{output}
...
""".strip()

    def build_improvement_prompt(self, original: str, critique: str, domain_text: str, df_columns: list[str]) -> str:
        return f"""
You are a data validation assistant.
...
### Dataset Columns

[{", ".join(df_columns)}]

...
{domain_text}
...
{critique}
...
{original}
...
""".strip()

    def run_rci_pipeline(self, prompt: str, domain_text: str, df_columns: list[str]) -> dict:
        logger.info("Running RuleRCI pipeline")

        initial_output = self.llm.call(prompt)
        logger.debug("Initial output received:\n%s", initial_output)

        critique_prompt = self.build_critique_prompt(initial_output, domain_text, df_columns)
        critique = self.llm.call(critique_prompt)
        logger.debug("Critique received:\n%s", critique)

        improve_prompt = self.build_improvement_prompt(initial_output, critique, domain_text, df_columns)
        improved_output = self.llm.call(improve_prompt)
        logger.debug("Improved output received:\n%s", improved_output)

        return {
            "initial_output": initial_output,
            "critique": critique,
            "improved_output": improved_output
        }

import os
from data_quality_tool.llm.llm_client import LLMClient
from data_quality_tool.config.logging_config import get_logger

logger = get_logger()


class RuleNoteExplainer:
    def __init__(self, domain_path: str, model: str = None):
        """
        Initialize the RuleNoteExplainer with domain knowledge and LLM model.

        Args:
            domain_path (str): Path to the domain knowledge file.
            model (str, optional): The LLM model name. Defaults to the value set in the environment.
        """
        self.llm = LLMClient(model=model)
        self.domain_knowledge = self._load_domain(domain_path)

    def _load_domain(self, path: str) -> str:
        """
        Load domain knowledge from a text file.

        Args:
            path (str): Path to the domain knowledge file.

        Returns:
            str: Contents of the domain file.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                logger.info("Loaded domain knowledge from %s", path)
                return content
        except Exception as e:
            logger.error("Failed to load domain knowledge from %s: %s", path, e)
            return ""

    def explain(self, entry: dict, entry_type: str = "rule") -> str:
        """
        Generate an explanation for a rule or note using LLM.

        Args:
            entry (dict): Dictionary containing the rule/note with keys like "id", "message"/"description", "code".
            entry_type (str): Either "rule" or "note".

        Returns:
            str: LLM-generated explanation.
        """
        identifier = entry.get("id", "unknown")
        description = entry.get("message") or entry.get("description", "")
        logic = entry.get("code", "")

        user_prompt = f"""
            You are given domain knowledge and a data quality {entry_type}.
            
            ### {entry_type.capitalize()} ID
            {identifier}
            
            ### Description
            {description}
            
            ### Logic
            {logic}
            
            ###Domain Knowledge
            {self.domain_knowledge}
            
            Explain clearly where this {entry_type} likely comes from, what it is intended to detect, and why it makes sense in this domain.
            Write the explanation in non-technical language suitable for business users.
            """
        try:
            logger.info("Generating explanation for %s '%s'", entry_type, identifier)
            response = self.llm.call(messages=[
                {"role": "system", "content": "You are an expert in data quality with a talent for explaining logic simply."},
                {"role": "user", "content": user_prompt}
            ])
            return response.strip()
        except Exception as e:
            logger.error("LLM failed to explain %s '%s': %s", entry_type, identifier, e)
            return f"Could not generate explanation due to LLM error."
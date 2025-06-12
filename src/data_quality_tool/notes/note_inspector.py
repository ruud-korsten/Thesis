import pandas as pd

from data_quality_tool.config.logging_config import get_logger

logger = get_logger()


def inspect_note_violations(df: pd.DataFrame, results: dict):
    valid_notes = [note for note in results if "violations" in results[note]]

    if not valid_notes:
        logger.info("No valid note violations found.")
        return

    logger.info("Available note checks:")
    for i, note in enumerate(valid_notes):
        logger.info("  %d. %s (%d violations)", i + 1, note, results[note]['violations'])

    # For interactive use, replace this part with actual user input in CLI/streamlit/etc.
    logger.warning("Interactive selection removed for logging version. Implement selection logic as needed.")
    # Example placeholder (e.g., auto-inspect the first note):
    try:
        idx = 0
        note = valid_notes[idx]
        func_name = results[note]['function_name']
        code = results[note]['code']

        logger.info("Inspecting function: %s", func_name)
        logger.debug("Code:\n%s", code)

        # Re-execute function to get mask
        local_env = {}
        exec(code, {"pd": pd}, local_env)
        func = next(v for v in local_env.values() if callable(v))
        mask = func(df)

        logger.info("First 5 violations for '%s':", note)
        logger.debug("\n%s", df[mask].head().to_string(index=False))
    except Exception as e:
        logger.exception("Error during inspection: %s", str(e))

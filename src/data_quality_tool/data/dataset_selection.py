import os
import pandas as pd

from .db import fetch_table
from data_quality_tool.logging_config import get_logger
from .dq_injection import HealthcareDQInjector, RetailDQInjector  # ✅ Correct imports

logger = get_logger()

try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    hf_load_dataset = None
    logger.warning("Hugging Face 'datasets' library is not installed.")

DATA_ROOT = "data"
DOMAIN_ROOT = "domain_knowledge"

# === Dataset Loader Registry ===
DATASET_LOADERS = {
    "retail": lambda: load_local_excel("retail", "retail.xlsx"),
    "energy": lambda: load_postgres_table("public_grafana_ovvia.energy_highfrequent", "test", "energy.txt"),
    "syrinx": lambda: load_postgres_table("public_ovvia.syrinx_measurements", "test", "syrinx.txt"),
    "sensor": lambda: load_huggingface_dataset("jeorgexyz/hvac_sensor_data", "sensor.txt"),
    "wallmart": lambda: load_huggingface_dataset("large-traversaal/Walmart-sales", "wallmart.txt"),
    "health": lambda: load_huggingface_dataset("vrajakishore/dummy_health_data", "health.txt"),
    "mqtt": lambda: load_postgres_table("public.mqtt_aggregate", "production", "mqtt.txt"),
}

# === DQ Injection Class Registry ===
DQ_INJECTORS = {
    "health": HealthcareDQInjector,
    "retail": RetailDQInjector,
    # Add other dataset-specific injectors here
}


def load_dataset(name: str, dirty: bool = False, output_format: str = "xlsx") -> tuple[pd.DataFrame, str]:
    logger.info("Loading dataset: %s (dirty=%s)", name, dirty)
    domain_path = os.path.join(DOMAIN_ROOT, f"{name}.txt")

    try:
        if name not in DATASET_LOADERS:
            logger.error("Unsupported dataset name: %s", name)
            raise ValueError(f"Unsupported dataset name: {name}")

        df, _ = DATASET_LOADERS[name]()

        if dirty:
            injector_class = DQ_INJECTORS.get(name)
            if injector_class:
                injector = injector_class()
                output_dir = f"./data/{name}/dq_injected_output"
                injector.inject_errors(df, output_dir, file_prefix=name)
                # Reload the modified dataset after injection
                df = _reload_dirty_dataset(output_dir, name, output_format)
            else:
                logger.warning("No DQ injector registered for dataset '%s'. Skipping injection.", name)

    except Exception as e:
        logger.exception("Failed to load dataset: %s", name)
        raise

    logger.info("Dataset '%s' loaded successfully with shape %s", name, df.shape)
    return df, domain_path


def _reload_dirty_dataset(output_dir: str, name: str, output_format: str) -> pd.DataFrame:
    file_extension = "csv" if output_format == "csv" else "xlsx"
    dirty_file = os.path.join(output_dir, f"{name}_dirty.{file_extension}")
    logger.info("Reloading dataset from injected dirty file: %s", dirty_file)
    if output_format == "csv":
        return pd.read_csv(dirty_file)
    return pd.read_excel(dirty_file)


def load_local_excel(folder: str, filename: str) -> tuple[pd.DataFrame, str]:
    path = os.path.join(DATA_ROOT, folder, filename)
    domain_path = os.path.join(DOMAIN_ROOT, f"{folder}.txt")
    logger.debug("Loading Excel file from %s", path)
    df = pd.read_excel(path)
    return df, domain_path


def load_local_csv(folder: str, filename: str) -> tuple[pd.DataFrame, str]:
    path = os.path.join(DATA_ROOT, folder, filename)
    domain_path = os.path.join(DOMAIN_ROOT, f"{folder}.txt")
    logger.debug("Loading CSV file from %s", path)
    df = pd.read_csv(path)
    return df, domain_path


def load_postgres_table(table_name: str, db_type: str, domain_txt: str) -> tuple[pd.DataFrame, str]:
    logger.debug("Fetching table %s from %s DB", table_name, db_type)
    df = fetch_table(table_name, db_type)
    domain_path = os.path.join(DOMAIN_ROOT, domain_txt)
    return df, domain_path


def load_huggingface_dataset(hf_name: str, domain_file: str = None) -> tuple[pd.DataFrame, str]:
    if hf_load_dataset is None:
        logger.error("Hugging Face 'datasets' library not available.")
        raise ImportError("Hugging Face 'datasets' library is not installed.")
    logger.debug("Loading Hugging Face dataset: %s", hf_name)
    ds = hf_load_dataset(hf_name)
    df = ds["train"].to_pandas()
    domain_path = os.path.join(DOMAIN_ROOT, domain_file or "domain.txt")
    return df, domain_path

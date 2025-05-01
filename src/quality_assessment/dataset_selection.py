import os
import pandas as pd
from .db import fetch_table

try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    hf_load_dataset = None

DATA_ROOT = "data"
DOMAIN_ROOT = "domain_files"

def load_dataset(name: str) -> tuple[pd.DataFrame, str]:
    domain_path = os.path.join(DOMAIN_ROOT, f"{name}.txt")

    if name == "retail":
        df, _ = load_local_excel("online_retail", "online_retail.xlsx")
    elif name == "energy":
        df, _ = load_postgres_table("public_grafana_ovvia.energy_highfrequent", "energy.txt")
    elif name == "syrinx":
        df, _ = load_postgres_table("public_ovvia.syrinx_measurements", "syrinx.txt")
    elif name == "sensor":
        df, _ = load_huggingface_dataset("jeorgexyz/hvac_sensor_data", "sensor.txt")
    elif name == "wallmart":
        df, _ = load_huggingface_dataset("large-traversaal/Walmart-sales", "wallmart.txt")
    elif name == "health":
        df, _ = load_huggingface_dataset("vrajakishore/dummy_health_data", "health.txt")
    else:
        raise ValueError(f"Unsupported dataset name: {name}")

    return df, domain_path

# --- Loaders ---

def load_local_excel(folder: str, filename: str) -> tuple[pd.DataFrame, str]:
    path = os.path.join(DATA_ROOT, folder, filename)
    domain_path = os.path.join(DOMAIN_ROOT, f"{folder}.txt")
    df = pd.read_excel(path)
    return df, domain_path

def load_local_csv(folder: str, filename: str) -> tuple[pd.DataFrame, str]:
    path = os.path.join(DATA_ROOT, folder, filename)
    domain_path = os.path.join(DOMAIN_ROOT, f"{folder}.txt")
    df = pd.read_csv(path)
    return df, domain_path

def load_postgres_table(table_name: str, domain_txt: str) -> tuple[pd.DataFrame, str]:
    df = fetch_table(table_name)
    domain_path = os.path.join(DOMAIN_ROOT, domain_txt)
    return df, domain_path

def load_huggingface_dataset(hf_name: str, domain_file: str = None) -> tuple[pd.DataFrame, str]:
    if hf_load_dataset is None:
        raise ImportError("Hugging Face 'datasets' library is not installed.")
    ds = hf_load_dataset(hf_name)
    df = ds["train"].to_pandas()
    domain_path = os.path.join(DOMAIN_ROOT, domain_file or "domain.txt")
    return df, domain_path


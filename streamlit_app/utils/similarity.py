import os
import pandas as pd
from glob import glob
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np

def compute_similarity_scores(domain_path, kind="rules", mode="previous", limit=None):
    run_paths = sorted(glob(os.path.join(domain_path, "*")))

    # Limit to last N runs if specified
    if limit is not None and limit > 0:
        run_paths = run_paths[-limit:]

    vectors = []
    labels = []

    for run_path in run_paths:
        run_id = os.path.basename(run_path)

        if kind == "rules":
            file_path = os.path.join(run_path, "rule_summary.csv")
            id_col = "Rule ID"
        elif kind == "notes":
            file_path = os.path.join(run_path, "note_summary.csv")
            id_col = "Function"
        else:
            raise ValueError("kind must be 'rules' or 'notes'")

        if not os.path.exists(file_path):
            continue

        try:
            df = pd.read_csv(file_path)
            df = df.set_index(id_col)["Violations"]
            df = pd.to_numeric(df, errors='coerce').fillna(0)
            vectors.append(df)
            labels.append(run_id)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")

    if len(vectors) < 2:
        return pd.DataFrame(columns=["Run ID", "Similarity (%)"])

    aligned_df = pd.DataFrame(vectors, index=labels).fillna(0)
    normed = normalize(aligned_df)

    scores = []
    for i in range(len(normed)):
        if i == 0:
            scores.append(None)
        else:
            ref_index = i - 1 if mode == "previous" else 0
            sim = cosine_similarity([normed[i]], [normed[ref_index]])[0][0]
            scores.append(round(sim * 100, 2))

    return pd.DataFrame({
        "Run ID": labels,
        "Similarity to Previous Run (%)" if mode == "previous" else "Similarity to First Run (%)": scores
    })


def compute_prediction_mask_similarity_scores(domain_path, mode="previous", limit=None):
    run_paths = sorted(glob(os.path.join(domain_path, "*")))

    if limit is not None and limit > 0:
        run_paths = run_paths[-limit:]

    vectors = []
    labels = []

    def flatten_mask(mask_df):
        binary = mask_df.notna().astype(int)
        stacked = binary.stack(dropna=False)
        stacked.index = stacked.index.map(lambda idx: f"{idx[0]}::{idx[1]}")
        return stacked

    for run_path in run_paths:
        run_id = os.path.basename(run_path)
        mask_path = os.path.join(run_path, "prediction_mask.csv")
        if not os.path.exists(mask_path):
            continue

        try:
            df = pd.read_csv(mask_path)
            vec = flatten_mask(df)
            print(f"[DEBUG] {run_id} → flattened mask with {len(vec)} cells")
            vectors.append(vec)
            labels.append(run_id)
        except Exception as e:
            print(f"Error processing mask for run {run_id}: {e}")

    if len(vectors) < 2:
        print("[DEBUG] Not enough valid runs for similarity comparison.")
        return pd.DataFrame(columns=["Run ID", "Similarity (%)"])

    aligned_df = pd.DataFrame(vectors, index=labels).fillna(0)
    print(f"[DEBUG] Aligned DataFrame shape: {aligned_df.shape}")
    print(f"[DEBUG] Aligned columns (features): {len(aligned_df.columns)}")

    normed = normalize(aligned_df)
    print(f"[DEBUG] Normalized matrix shape: {normed.shape}")

    scores = []
    for i in range(len(normed)):
        if i == 0:
            scores.append(None)
        else:
            ref_index = i - 1 if mode == "previous" else 0
            sim = cosine_similarity([normed[i]], [normed[ref_index]])[0][0]
            print(f"[DEBUG] Comparing Run {labels[i]} to Run {labels[ref_index]} → cosine: {sim:.5f}")
            scores.append(round(sim * 100, 2))

    return pd.DataFrame({
        "Run ID": labels,
        "Similarity to Previous Run (%)" if mode == "previous" else "Similarity to First Run (%)": scores
    })
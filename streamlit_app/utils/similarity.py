import os
import pandas as pd
from glob import glob
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
import json
import hashlib
import re
from difflib import SequenceMatcher
import streamlit as st

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

# ========== Note Change Tracker ==========

def normalize_code(code):
    code = re.sub(r"#.*", "", code)
    code = re.sub(r"\s+", "", code)
    return code.strip()

def hash_code(code):
    return hashlib.md5(normalize_code(code).encode()).hexdigest()

def desc_similarity(desc1, desc2):
    return SequenceMatcher(None, desc1, desc2).ratio()

def track_note_function_changes_across_runs(run_paths):
    all_notes = []
    hash_lookup = {}

    for run_path in run_paths:
        run_id = os.path.basename(run_path)
        file_path = os.path.join(run_path, "note_functions.json")
        if not os.path.exists(file_path):
            continue

        try:
            with open(file_path, encoding="utf-8") as f:
                notes_dict = json.load(f)

            for i, (description, code) in enumerate(notes_dict.items()):
                code_hash = hash_code(code)
                all_notes.append({
                    "run": run_id,
                    "note_id": f"N{i + 1:03}",
                    "hash": code_hash,
                    "description": description
                })

                if code_hash not in hash_lookup:
                    hash_lookup[code_hash] = []
                hash_lookup[code_hash].append(run_id)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    df = pd.DataFrame(all_notes)
    pivot = df.pivot_table(index="hash", columns="run", values="note_id", aggfunc="first")

    hash_stats = df.groupby("hash").agg(
        runs_seen=("run", "nunique"),
        example_description=("description", "first")
    ).reset_index()

    hash_stats["status"] = hash_stats["runs_seen"].apply(
        lambda x: "stable" if x == len(run_paths) else ("intermittent" if x > 1 else "unique")
    )

    return {
        "note_history": pivot,
        "change_summary": hash_stats.value_counts("status").to_dict(),
        "note_stats": hash_stats
    }

# ========== Optional Streamlit Display ==========

def display_note_change_overview(run_paths):
    result = track_note_function_changes_across_runs(run_paths)
    note_history = result["note_history"]
    change_summary = result["change_summary"]
    note_stats = result["note_stats"]

    st.markdown("### 🧠 Note Function Change Summary")
    st.write({
        "Total Unique Notes (by logic)": len(note_stats),
        **change_summary
    })

    color_map = {"stable": "✅", "intermittent": "🟡", "unique": "🆕"}
    note_stats["Status Icon"] = note_stats["status"].map(color_map)
    st.dataframe(note_stats[["Status Icon", "status", "runs_seen", "example_description"]])

    with st.expander("📋 Note Appearance Matrix (hash × run)"):
        st.dataframe(note_history.fillna(""))

    return result

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def add_semantic_note_clusters(note_stats, threshold=0.9):
    """
    Groups notes with similar descriptions into semantic clusters.
    Adds a `cluster_id` column to the note_stats dataframe.
    """
    descriptions = note_stats["example_description"].tolist()
    vectorizer = TfidfVectorizer().fit_transform(descriptions)
    sim_matrix = cosine_similarity(vectorizer)

    cluster_ids = [-1] * len(descriptions)
    cluster_counter = 0

    for i in range(len(descriptions)):
        if cluster_ids[i] == -1:
            cluster_ids[i] = f"G{cluster_counter:03}"
            for j in range(i + 1, len(descriptions)):
                if sim_matrix[i, j] >= threshold:
                    cluster_ids[j] = f"G{cluster_counter:03}"
            cluster_counter += 1

    note_stats["cluster_id"] = cluster_ids
    return note_stats

# Test dry-run with available note_stats
sample_note_stats = pd.DataFrame({
    "example_description": [
        "Patient_ID is numeric, so should be checked for outliers.",
        "Patient_ID is numeric and should be checked for outliers.",
        "Gender must be either Male or Female.",
        "Check if Patient_ID is an outlier",
        "Validate if gender is Male or Female",
        "Age must be a positive number.",
    ]
})

add_semantic_note_clusters(sample_note_stats)

def hash_rule(rule):
    rule_normalized = {
        "id": rule.get("id", ""),
        "name": rule.get("name", ""),
        "type": rule.get("type", ""),
        "column": rule.get("column", ""),
        "condition": rule.get("condition", {}),
        "message": rule.get("message", "")
    }
    rule_str = json.dumps(rule_normalized, sort_keys=True)
    return hashlib.md5(rule_str.encode("utf-8")).hexdigest()

def track_rule_changes_across_runs(run_paths):
    rule_records = []

    for run_path in run_paths:
        run_id = os.path.basename(run_path)
        rule_file = os.path.join(run_path, "rules.json")

        if not os.path.exists(rule_file):
            continue

        try:
            with open(rule_file, encoding="utf-8") as f:
                rules = json.load(f)

            for i, rule in enumerate(rules):
                rule_hash = hash_rule(rule)
                rule_records.append({
                    "run": run_id,
                    "rule_id": rule.get("id", f"R{i + 1:03}"),
                    "hash": rule_hash,
                    "description": rule.get("message", ""),
                    "column": rule.get("column", ""),
                    "type": rule.get("type", "")
                })

        except Exception as e:
            print(f"[DEBUG] Failed to parse {rule_file}: {e}")

    if not rule_records:
        return None

    df = pd.DataFrame(rule_records)
    rule_summary = df.groupby("hash").agg({
        "rule_id": "first",
        "description": "first",
        "column": "first",
        "type": "first",
        "run": "count"
    }).rename(columns={"run": "runs_seen"}).reset_index()

    rule_matrix = df.pivot(index="hash", columns="run", values="rule_id").fillna("")
    return {
        "rule_stats": rule_summary,
        "rule_history": rule_matrix
    }

# Attempt to test using available sample run paths (mocking the user's health dataset)
sample_health_run_paths = sorted(glob("artifacts/runs/health/*"))
track_rule_changes_across_runs(sample_health_run_paths)
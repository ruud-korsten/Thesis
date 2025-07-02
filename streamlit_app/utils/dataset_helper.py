import pandas as pd
import streamlit as st
import colorsys
import random
import hashlib


def render_dataset_with_slider(dataset_path: str, max_display_rows: int = 1000):
    """
    Render a dataset with a Streamlit slider to control number of displayed rows.

    Args:
    - dataset_path (str): Path to the CSV dataset.
    - max_display_rows (int): Maximum rows to display for performance.
    """
    try:
        dataset = pd.read_csv(dataset_path)
        total_rows = dataset.shape[0]

        row_count = st.slider(
            "Number of rows to display:",
            min_value=1,
            max_value=min(max_display_rows, total_rows),
            value=10,
        )
        st.dataframe(dataset.head(row_count), use_container_width=True)
    except FileNotFoundError:
        st.error("Dataset not found! Please verify the path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

def filter_and_highlight(df, mask_df, rules, notes,
                         missing=False, duplicates=False, type_mismatch=False,
                         row_limit=None):

    filtered_rows = pd.Series(False, index=df.index)
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    text_color = "color: black"

    # --- Rule Matching ---
    for rule in rules:
        if rule["column"] in mask_df.columns:
            rule_mask = mask_df[rule["column"]].astype(str).str.contains(rule["id"], na=False)
            print(f"[RULE] {rule['id']} → column: {rule['column']}, matches: {rule_mask.sum()}")
            filtered_rows |= rule_mask
            color = rule.get("color", "#ffeb3b")
            styles.loc[rule_mask, rule["column"]] = f"background-color: {color}; {text_color}"
        else:
            print(f"[RULE WARNING] Column '{rule['column']}' not in mask_df.columns")

    # --- Note Matching ---
    for note in notes:
        note_id = note["id"]
        color = note.get("color", "#c8e6c9")

        # Match note ID anywhere in mask_df
        match = mask_df.applymap(lambda x: note_id in str(x) if pd.notnull(x) else False)
        note_mask = match.any(axis=1)
        filtered_rows |= note_mask

        for col in mask_df.columns:
            styles.loc[match[col], col] = f"background-color: {color}; color: black"

        print(f"[NOTE] {note_id} → matches: {note_mask.sum()} rows")

    # --- General Filters ---
    if missing:
        missing_cell_mask = df.isnull()
        filtered_rows |= missing_cell_mask.any(axis=1)
        styles = styles.mask(missing_cell_mask, "background-color: red; color: white")

    if duplicates:
        dup_mask = df.duplicated()
        filtered_rows |= dup_mask
        for col in df.columns:
            styles.loc[dup_mask, col] = "background-color: orange; color: black"

    if type_mismatch:
        type_mismatch_mask = mask_df.applymap(
            lambda x: isinstance(x, str) and "TYPE_MISMATCH" in x
        )
        tm_mask = type_mismatch_mask.any(axis=1)
        filtered_rows |= tm_mask
        for col in mask_df.columns:
            styles.loc[type_mismatch_mask[col], col] = "background-color: purple; color: white"

    print(f"[SUMMARY] Total filtered rows: {filtered_rows.sum()}")

    # --- Apply Filters ---
    filtered_df = df[filtered_rows]
    filtered_styles = styles.loc[filtered_rows]

    if row_limit:
        filtered_df = filtered_df.head(row_limit)
        filtered_styles = filtered_styles.head(row_limit)

    return filtered_df.style.apply(lambda _: filtered_styles, axis=None), filtered_rows


def generate_distinct_colors(ids, saturation=0.7, brightness=0.95, hue_steps=36):
    """
    Generate stable, visually distinct colors from IDs.
    - hue_steps = number of discrete hue buckets (more = finer spacing)
    """
    colors = []
    for identifier in ids:
        hash_int = int(hashlib.sha256(identifier.encode()).hexdigest(), 16)
        hue_index = hash_int % hue_steps
        hue = (hue_index / hue_steps)  # normalized 0-1
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, brightness)
        color = '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
        colors.append(color)
    return colors

import pandas as pd
import streamlit as st


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

def filter_and_highlight(df, mask_df, rules, notes, row_limit=None):
    filtered_rows = pd.Series(False, index=df.index)
    styles = pd.DataFrame('', index=df.index, columns=df.columns)

    color_map = {
        "rule": "#ffeb3b",
        "note": "#c8e6c9"
    }
    text_color = "color: black"

    for rule in rules:
        rule_mask = mask_df[rule["column"]] == rule["id"]
        filtered_rows |= rule_mask
        styles.loc[rule_mask, rule["column"]] = f"background-color: {color_map['rule']}; {text_color}"

    for note in notes:
        note_mask = mask_df[note["column"]] == note["id"]
        filtered_rows |= note_mask
        styles.loc[note_mask, note["column"]] = f"background-color: {color_map['note']}; {text_color}"

    filtered_df = df[filtered_rows]
    filtered_styles = styles.loc[filtered_rows]

    if row_limit:
        filtered_df = filtered_df.head(row_limit)
        filtered_styles = filtered_styles.head(row_limit)

    return filtered_df.style.apply(lambda _: filtered_styles, axis=None)
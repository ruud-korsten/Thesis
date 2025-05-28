import streamlit as st
import pandas as pd
from utils.dataset_helper import filter_and_highlight
from utils.load_summary import (
    load_dataset_and_mask,
    load_summary_files,
    load_rule_and_note_definitions,
)
from utils.session_state_helper import clear_all_filters


def render_summary_tab(summary, selected_display_name, dq_card):
    st.title(f"Data Quality Report: {selected_display_name}")

    overview = summary["overview"]
    cols = st.columns(2)
    with cols[0].expander("Dataset Overview", expanded=True):
        st.markdown(f"- **Shape**: `{overview['shape'][0]} rows × {overview['shape'][1]} columns`")
        st.markdown(f"- **Columns**: `{', '.join(overview['columns'])}`")
        st.markdown("### Sample Rows")
        st.dataframe(overview["sample_rows"], use_container_width=True)
        st.markdown("### Summary Statistics")
        st.dataframe(overview["summary_stats"], use_container_width=True)

    missing = summary["missing"]
    dq_card(cols[1], "Missing Values", {
        "passed": missing["missing_values"].sum() == 0,
        "summary": f"{(missing['missing_values'] > 0).sum()} columns with missing data",
        "details": [
            f"`{col}` → {row['missing_values']} missing ({row['percent_missing']:.2f}%)"
            for col, row in missing.iterrows()
        ]
    })

    dup = summary["duplicates"]
    dq_card(cols[0], "Duplicates", {
        "passed": dup["summary"]["duplicate_rows"] == 0,
        "summary": f"{dup['summary']['duplicate_rows']} rows duplicated",
        "details": [
            f"Total Rows: {dup['summary']['total_rows']}",
            f"Duplicate Rows: {dup['summary']['duplicate_rows']}",
            f"Percent: {dup['summary']['percent']}%"
        ]
    })
    with cols[0].expander("Duplicate Row Examples"):
        st.dataframe(dup["examples"], use_container_width=True)

    rules = summary["rules"]
    dq_card(cols[1], "Rule Violations", {
        "passed": rules["Violations"].sum() == 0,
        "summary": f"{(rules['Violations'] > 0).sum()} rules failed",
        "details": [
            f"`{row['Rule ID']}` → {row['Violations']} violations — {row['Description']}"
            for _, row in rules.iterrows() if row["Violations"] > 0
        ] or ["No rule violations."]
    })

    notes = summary["notes"]
    dq_card(cols[1], "Note Violations", {
        "passed": notes["Violations"].sum() == 0,
        "summary": f"{(notes['Violations'] > 0).sum()} notes triggered",
        "details": [
            f"`{row['Function']}` → {row['Violations']} hits — {row['Note']}"
            for _, row in notes.iterrows() if row["Violations"] > 0
        ] or ["No note violations."]
    })



def clear_all_filters(rules, notes):
    for rule in rules:
        st.session_state[f"checkbox_rule_{rule['id']}"] = False
    for note in notes:
        st.session_state[f"checkbox_note_{note['id']}"] = False
    st.session_state["checkbox_missing"] = False
    st.session_state["checkbox_duplicates"] = False

def render_detailed_tab_with_interactions(run_path):
    st.title("Detailed Report")

    try:
        dataset, pred_mask = load_dataset_and_mask(run_path)
        rule_summary, note_summary = load_summary_files(run_path)
        rule_defs, note_descs = load_rule_and_note_definitions(run_path)
    except RuntimeError as e:
        st.error(str(e))
        return

    rules = []
    for rule in rule_defs:
        summary_row = rule_summary[rule_summary["Rule ID"] == rule["id"]]
        violations = int(summary_row["Violations"].values[0]) if not summary_row.empty else 0
        rules.append({
            "id": rule["id"],
            "name": rule.get("name", ""),
            "description": rule.get("message", ""),
            "column": rule.get("column", ""),
            "violations": violations
        })

    notes = []
    for i, desc in enumerate(note_descs):
        note_id = f"N{i+1:03}"
        summary_row = note_summary.iloc[i] if i < len(note_summary) else {}
        violations = int(summary_row["Violations"]) if "Violations" in summary_row else 0
        column = next((col for col in dataset.columns if col.lower() in desc.lower()), None)
        notes.append({
            "id": note_id,
            "name": desc[:40] + "..." if len(desc) > 40 else desc,
            "description": desc,
            "column": column,
            "violations": violations
        })

    if "show_all_rules" not in st.session_state:
        st.session_state["show_all_rules"] = False
    if "show_all_notes" not in st.session_state:
        st.session_state["show_all_notes"] = False

    cols = st.columns([3, 1])
    with cols[1]:
        st.header("Filters")

        if st.button("Clear All Filters"):
            clear_all_filters(rules, notes)

        st.markdown("---")

        st.subheader("Rules")
        visible_rule_count = len([r for r in rules if r["violations"] > 0])
        total_rule_count = len(rules)
        st.checkbox(f"Display all rules ({visible_rule_count} of {total_rule_count} shown)", key="show_all_rules")
        visible_rules = rules if st.session_state["show_all_rules"] else [r for r in rules if r["violations"] > 0]
        for rule in visible_rules:
            key = f"checkbox_rule_{rule['id']}"
            label = f"{rule['id']}: {rule['name']} ({rule['violations']} violations)"
            st.checkbox(label, key=key)
        st.markdown("---")

        st.subheader("Notes")
        visible_note_count = len([n for n in notes if n["violations"] > 0])
        total_note_count = len(notes)
        st.checkbox(f"Display all notes ({visible_note_count} of {total_note_count} shown)", key="show_all_notes")
        visible_notes = notes if st.session_state["show_all_notes"] else [n for n in notes if n["violations"] > 0]
        for note in visible_notes:
            key = f"checkbox_note_{note['id']}"
            label = f"{note['id']}: {note['name']} ({note['violations']} violations)"
            st.checkbox(label, key=key)
        st.markdown("---")

        st.subheader("General Filters")
        st.checkbox("Missing Values", key="checkbox_missing")
        st.checkbox("Duplicates", key="checkbox_duplicates")

    with cols[0]:
        st.header("Dataset Viewer")

        active_rules = [r for r in rules if st.session_state.get(f"checkbox_rule_{r['id']}", False)]
        active_notes = [n for n in notes if st.session_state.get(f"checkbox_note_{n['id']}", False)]

        active_items = []
        if active_rules:
            active_items.extend([f"Rule: {r['id']} - {r['name']}" for r in active_rules])
        if active_notes:
            active_items.extend([f"Note: {n['id']} - {n['name']}" for n in active_notes])
        if st.session_state.get("checkbox_missing"):
            active_items.append("Missing Values")
        if st.session_state.get("checkbox_duplicates"):
            active_items.append("Duplicate Rows")

        st.markdown("**Active View:** " + (", ".join(active_items) if active_items else "_None_"))

        dataset_placeholder = st.empty()
        styled = None

        if active_rules or active_notes:
            styled = filter_and_highlight(dataset, pred_mask, active_rules, active_notes)
        elif st.session_state.get("checkbox_missing"):
            missing_rows = dataset.isnull().any(axis=1)
            styled = dataset[missing_rows].style.applymap(
                lambda x: "background-color: red; color: white",
                subset=pd.IndexSlice[:, dataset.columns[dataset.isnull().any()]]
            )
        elif st.session_state.get("checkbox_duplicates"):
            styled = dataset[dataset.duplicated()]

        if styled is not None:
            dataset_placeholder.dataframe(styled, use_container_width=True, height=800)
        else:
            dataset_placeholder.dataframe(dataset, use_container_width=True, height=800)

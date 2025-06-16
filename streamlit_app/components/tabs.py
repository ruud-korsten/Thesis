import streamlit as st
import pandas as pd
from utils.dataset_helper import (
    filter_and_highlight,
    generate_distinct_colors
)
from utils.load_summary import (
    load_dataset_and_mask,
    load_summary_files,
    load_rule_and_note_definitions,
)
from data_quality_tool.evaluation.final_validation import FinalValidator
from utils.session_state_helper import clear_all_filters
import sys
import os
from glob import glob
from utils.similarity import compute_similarity_scores, compute_prediction_mask_similarity_scores, track_note_function_changes_across_runs, display_note_change_overview, add_semantic_note_clusters, track_rule_changes_across_runs
import altair as alt
import os



print("PYTHONPATH:", sys.path)
print("Current dir:", os.getcwd())
print("Check file exists:", os.path.exists("src/data_quality_tool/evaluation/explain_violation.py"))
try:
    from data_quality_tool.evaluation.explain_violation import ExplainViolation
    print("ExplainViolation imported successfully")
except Exception as e:
    print("Import failed:", e)

def render_summary_tab(summary, selected_display_name, dq_card, selected_run):
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
    # --- Schema Mismatches ---
    schema = summary.get("schema_mismatches")
    if schema is not None and not schema.empty:
        dq_card(cols[0], "Schema Mismatches", {
            "passed": schema["count"].sum() == 0,
            "summary": f"{schema['count'].astype(bool).sum()} columns with mismatches",
            "details": [
                f"`{row['column']}` → {row['count']} {row['mismatch_type']}s "
                f"(expected: {row['expected_type']}, actual: {row['actual_type']})"
                for _, row in schema.iterrows()
            ]
        })
    else:
        dq_card(cols[0], "Schema Mismatches", {
            "passed": True,
            "summary": "No schema mismatches detected.",
            "details": []
        })

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
    notes["Violations"] = pd.to_numeric(notes["Violations"], errors="coerce").fillna(0)

    dq_card(cols[1], "Note Violations", {
        "passed": notes["Violations"].sum() == 0,
        "summary": f"{(notes['Violations'] > 0).sum()} notes triggered",
        "details": [
                       f"`{row['Function']}` → {int(row['Violations'])} hits — {row['Note']}"
                       for _, row in notes[notes["Violations"] > 0].iterrows()
                   ] or ["No note violations."]
    })

    # --- Final Validation Section ---
    with st.expander("Final Validation Feedback", expanded=False):
        try:
            # Point to the specific run folder and load feedback file
            validation_path = os.path.join(selected_run, "final_validation.txt")
            print("VALIDATION PATH ---------------> ",validation_path)
            if os.path.exists(validation_path):
                with open(validation_path, "r", encoding="utf-8") as f:
                    validation_feedback = f.read()

                st.markdown("**LLM Feedback on Rules & Notes:**")
                st.info(validation_feedback)
            else:
                st.warning("No final validation report found for this run.")

        except Exception as e:
            st.error(f"Final validation failed to load: {e}")


def render_detailed_tab_with_interactions(run_path):
    st.title("Detailed Report")

    try:
        dataset, pred_mask = load_dataset_and_mask(run_path)

        # Fix misalignment: drop index column if present
        if "Unnamed: 0" in dataset.columns:
            dataset = dataset.drop(columns=["Unnamed: 0"])

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
        note_id = f"N{i + 1:03}"
        summary_row = note_summary.iloc[i] if i < len(note_summary) else {}
        try:
            violations = int(summary_row["Violations"]) if "Violations" in summary_row else 0
        except (ValueError, TypeError):
            violations = 0

        # Try to match columns from description
        matched_columns = [col for col in dataset.columns if col.lower() in desc.lower()]

        # Expand match if "numeric" is mentioned or description implies general numeric handling
        if "numeric" in desc.lower() or "all numeric columns" in desc.lower():
            numeric_cols = dataset.select_dtypes(include="number").columns.tolist()
            matched_columns = list(set(matched_columns) | set(numeric_cols))  # merge with any existing matches

        # Debug output
        print(f"[DEBUG] Note {note_id}")
        print(f"  Description: {desc}")
        print(f"  Matched Columns: {matched_columns}")
        print(f"  Violations: {violations}")
        if not matched_columns:
            st.text(f"[DEBUG] No column match for note '{note_id}': {desc}")

        notes.append({
            "id": note_id,
            "name": desc[:40] + "..." if len(desc) > 40 else desc,
            "description": desc,
            "column": matched_columns if matched_columns else None,
            "violations": violations
        })

    # Collect all IDs first
    rule_ids = [rule["id"] for rule in rules]
    note_ids = [note["id"] for note in notes]
    all_ids = rule_ids + note_ids

    # Generate stable distinct colors for all
    all_colors = generate_distinct_colors(all_ids, hue_steps=72, saturation=0.75, brightness=0.95)

    # Assign back to rules
    rule_color_map = dict(zip(rule_ids, all_colors[:len(rule_ids)]))
    for rule in rules:
        rule["color"] = rule_color_map[rule["id"]]

    # Assign back to notes
    note_color_map = dict(zip(note_ids, all_colors[len(rule_ids):]))
    for note in notes:
        note["color"] = note_color_map[note["id"]]

    if "show_all_rules" not in st.session_state:
        st.session_state["show_all_rules"] = False
    if "show_all_notes" not in st.session_state:
        st.session_state["show_all_notes"] = False

    cols = st.columns([1, 3])
    with cols[0]:
        st.header("Filters")

        if st.button("Clear All Filters"):
            clear_all_filters(rules, notes)

        st.markdown("---")

        # --- Rules ---
        st.subheader("Rules")
        visible_rule_count = len([r for r in rules if r["violations"] > 0])
        total_rule_count = len(rules)
        st.checkbox(f"Display all rules ({visible_rule_count} of {total_rule_count} shown)", key="show_all_rules")

        visible_rules = rules if st.session_state["show_all_rules"] else [r for r in rules if r["violations"] > 0]

        for rule in visible_rules:
            key = f"checkbox_rule_{rule['id']}"
            checkbox_col, label_col = st.columns([1, 12])

            with checkbox_col:
                st.checkbox(" ", key=key, label_visibility="collapsed")

            with label_col:
                dot = (
                    f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;"
                    f"background-color:{rule['color']};margin-right:8px;vertical-align:middle;'></span>"
                )
                label_html = (
                    f"<div style='display:flex;align-items:center;'>"
                    f"{dot}<span title='{rule['description']}'>{rule['id']}: {rule['name']} "
                    f"({rule['violations']} violations)</span></div>"
                )
                st.markdown(label_html, unsafe_allow_html=True)

        st.markdown("---")

        # --- Notes ---
        st.subheader("Notes")
        visible_note_count = len([n for n in notes if n["violations"] > 0])
        total_note_count = len(notes)
        st.checkbox(f"Display all notes ({visible_note_count} of {total_note_count} shown)", key="show_all_notes")

        visible_notes = notes if st.session_state["show_all_notes"] else [n for n in notes if n["violations"] > 0]

        for note in visible_notes:
            key = f"checkbox_note_{note['id']}"
            checkbox_col, label_col = st.columns([1, 12])

            with checkbox_col:
                st.checkbox(" ", key=key, label_visibility="collapsed")

            with label_col:
                dot = (
                    f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;"
                    f"background-color:{note['color']};margin-right:8px;vertical-align:middle;'></span>"
                )
                label_html = (
                    f"<div style='display:flex;align-items:center;'>"
                    f"{dot}<span title='{note['description']}'>{note['id']}: {note['name']} "
                    f"({note['violations']} violations)</span></div>"
                )
                st.markdown(label_html, unsafe_allow_html=True)

        st.markdown("---")

        st.subheader("General Filters")
        st.checkbox("Missing Values", key="checkbox_missing")
        st.checkbox("Duplicates", key="checkbox_duplicates")
        st.checkbox("Type Mismatches", key="checkbox_type_mismatch")

    with cols[1]:
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

        print("Dataset columns:", dataset.columns.tolist())
        print("Prediction mask columns:", pred_mask.columns.tolist())

        if active_rules or active_notes:
            styled, filtered_rows = filter_and_highlight(dataset, pred_mask, active_rules, active_notes)
            display_df = dataset[filtered_rows]

        elif st.session_state.get("checkbox_missing"):
            display_df = dataset[dataset.isnull().any(axis=1)]

            def highlight_missing(val):
                return "background-color: red; color: white" if pd.isnull(val) else ""

            styled = display_df.style.applymap(highlight_missing)

        elif st.session_state.get("checkbox_duplicates"):
            display_df = dataset[dataset.duplicated()]
            styled = display_df.style.applymap(
                lambda x: "background-color: orange; color: black"
            )

        elif st.session_state.get("checkbox_type_mismatch"):
            type_mismatch_mask = pred_mask.applymap(
                lambda x: isinstance(x, str) and "TYPE_MISMATCH" in x
            )
            display_df = dataset[type_mismatch_mask.any(axis=1)]

            def highlight_type_mismatch(val, is_mismatch):
                return "background-color: purple; color: white" if is_mismatch else ""

            styled = display_df.style.apply(
                lambda row: [
                    highlight_type_mismatch(val, type_mismatch_mask.loc[row.name, col])
                    for col, val in row.items()
                ],
                axis=1
            )

        else:
            display_df = dataset
            styled = None  # fallback: unstyled full dataset

        # --- Display the styled DataFrame with pagination ---
        if styled is not None:
            dataset_placeholder.dataframe(styled, use_container_width=True, height=800)
        else:
            st.data_editor(display_df.head(10000), use_container_width=True, num_rows="fixed")


        # --- Row selection (shown only for filtered display) ---
        selected_index = None
        if not display_df.empty:
            selected_index = st.selectbox(
                "Select a row with issues to generate a report:",
                options=display_df.index.tolist(),
                format_func=lambda i: f"Row {i}"
            )

        # --- Explain violation Button ---
        if st.button("Explain violation", disabled=(selected_index is None)):
            dataset_name = os.path.basename(os.path.dirname(run_path))
            run_timestamp = os.path.basename(run_path)
            explainer = ExplainViolation(dataset_name=dataset_name, run_timestamp=run_timestamp)
            explanation = explainer.explain_violation(selected_index)
            st.markdown("### Violation Explanation")
            st.info(explanation)

        if st.button("Root Cause explanation", disabled=(selected_index is None)):
            dataset_name = os.path.basename(os.path.dirname(run_path))
            run_timestamp = os.path.basename(run_path)
            explainer = ExplainViolation(dataset_name=dataset_name, run_timestamp=run_timestamp)
            explanation = explainer.explain_violation_with_history(selected_index)
            st.markdown("### Root cause explanation")
            st.info(explanation)

def render_historical_tab(domain_path):
    st.title("Historical Data Quality Trends")

    # --- Load all runs and validate ---
    all_runs = sorted(glob(os.path.join(domain_path, "*")))
    if not all_runs:
        st.warning("No runs found in this domain.")
        return

    if len(all_runs) < 2:
        st.info("Not enough runs to analyze trends. At least 2 required.")
        return

    # --- Slider to choose how many runs to view ---
    default_limit = min(10, len(all_runs))
    control_col, _ = st.columns([2, 8])  # visually compact slider
    if len(all_runs) < 3:
        st.info(f"Only {len(all_runs)} valid run(s) available. Showing all.")
        num_runs_to_show = len(all_runs)
    else:
        num_runs_to_show = st.slider(
            "Number of runs",
            min_value=2,
            max_value=len(all_runs),
            value=default_limit,
            step=1,
            help="Select how many recent runs to display"
        )

    run_data = []
    run_paths = all_runs[-num_runs_to_show:]

    # --- Overview Summary ---
    st.markdown("### 📊 Overview Summary")
    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric(label="Total Runs", value=f"{len(run_paths)}")
    with summary_cols[1]:
        st.metric(label="Dataset", value=os.path.basename(domain_path))

    for run_path in run_paths:
        timestamp = os.path.basename(run_path)

        try:
            rule_path = os.path.join(run_path, "rule_summary.csv")
            note_path = os.path.join(run_path, "note_summary.csv")
            mask_path = os.path.join(run_path, "prediction_mask.csv")
            data_path = os.path.join(run_path, "dataset.csv")

            for file in [rule_path, note_path, mask_path, data_path]:
                if not os.path.exists(file) or os.path.getsize(file) == 0:
                    raise ValueError(f"Missing or empty file: {os.path.basename(file)}")

            rules_df = pd.read_csv(rule_path)
            notes_df = pd.read_csv(note_path)
            dataset = pd.read_csv(data_path)
            mask = pd.read_csv(mask_path)

            total_rule_violations = rules_df["Violations"].sum()
            total_note_violations = pd.to_numeric(notes_df["Violations"], errors="coerce").fillna(0).sum()
            total_violations = total_rule_violations + total_note_violations
            missing_values = dataset.isnull().sum().sum()
            type_mismatches = mask.applymap(lambda x: isinstance(x, str) and "TYPE_MISMATCH" in x).sum().sum()
            duplicate_rows = dataset.duplicated().sum()

            run_data.append({
                "run_id": timestamp,
                "total_rule_violations": total_rule_violations,
                "total_note_violations": total_note_violations,
                "total_violations": total_violations,
                "missing_values": missing_values,
                "type_mismatches": type_mismatches,
                "duplicate_rows": duplicate_rows
            })

        except Exception as e:
            st.warning(f"Error processing run {timestamp}: {e}")
            continue

    if not run_data:
        st.info("No valid data found in selected runs.")
        return

    df = pd.DataFrame(run_data).sort_values("run_id")
    df["run_index"] = list(range(1, len(df) + 1))
    df["run_label"] = df["run_index"].apply(lambda i: f"Run {i}")

    # --- Most Violated Rule ---
    most_common_rule = None
    max_violations = 0
    for run_path in run_paths:
        try:
            rule_summary_path = os.path.join(run_path, "rule_summary.csv")
            if os.path.exists(rule_summary_path) and os.path.getsize(rule_summary_path) > 0:
                rules_df = pd.read_csv(rule_summary_path)
                top_rule = rules_df.loc[rules_df["Violations"].idxmax()]
                if top_rule["Violations"] > max_violations:
                    max_violations = top_rule["Violations"]
                    most_common_rule = f"{top_rule['Rule ID']} ({int(top_rule['Violations'])} hits)"
        except Exception:
            continue

    with summary_cols[2]:
        st.metric(label="Most Violated Rule", value=most_common_rule or "N/A")

    # --- Altair Trend Chart ---
    st.subheader("Violation Trends Across Runs")
    melted_df = df.melt(
        id_vars=["run_index"],
        value_vars=[
            "total_rule_violations", "total_note_violations", "total_violations",
            "missing_values", "type_mismatches", "duplicate_rows"
        ],
        var_name="Violation Type",
        value_name="Count"
    )

    chart = alt.Chart(melted_df).mark_line(point=True).encode(
        x=alt.X("run_index:O", title="Run Number", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Count:Q", title="Violation Count"),
        color=alt.Color("Violation Type:N", legend=alt.Legend(title="Violation Type")),
        tooltip=["run_index", "Violation Type", "Count"]
    ).properties(
        width="container",
        height=400
    )

    st.altair_chart(chart, use_container_width=True)

    # --- Raw Table Display ---
    with st.expander("Run Metadata and Issue Totals"):
        st.dataframe(df[[
            "run_label", "run_id", "total_rule_violations", "total_note_violations",
            "total_violations", "missing_values", "type_mismatches", "duplicate_rows"
        ]], use_container_width=True)

    # --- Similarity Score Block ---
    sim_df = compute_prediction_mask_similarity_scores(
        domain_path,
        mode="previous",
        limit=num_runs_to_show
    )

    if not sim_df.empty:
        valid_scores = sim_df.dropna().tail(num_runs_to_show)
        if not valid_scores.empty:
            avg_score = valid_scores.iloc[:, 1].mean()
            norm_score = avg_score / 100
            blue_intensity = int(255 * norm_score)
            bg_color = f"rgb({255 - blue_intensity}, {255 - blue_intensity}, 255)"

            drift_level = (
                "✅ Good" if avg_score > 85 else
                "⚠️ Moderate" if avg_score > 65 else
                "🚨 High Drift"
            )

            st.markdown(f"### Prediction Similarity (Last {num_runs_to_show} Runs)")
            st.markdown(
                f"""
                <div style="
                    background-color: {bg_color};
                    color: black;
                    padding: 2rem;
                    font-size: 2rem;
                    font-weight: bold;
                    text-align: center;
                    border-radius: 10px;
                    border: 1px solid #ccc;
                    ">
                    {avg_score:.2f}%<br>
                    <span style='font-size: 1rem; font-weight: normal;'>Drift Level: {drift_level}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    # --- Note Drift Analysis ---
    st.subheader("🧠 Note Function Drift Across Runs")
    try:

        result = track_note_function_changes_across_runs(run_paths)
        note_stats = result["note_stats"]
        note_stats = add_semantic_note_clusters(note_stats)

        st.markdown("### Note Function Change Summary")
        st.write({
            "Total Unique Notes (by logic)": len(note_stats),
            **result["change_summary"]
        })

        color_map = {"stable": "✅", "intermittent": "🟡", "unique": "🆕"}
        note_stats["Status Icon"] = note_stats["status"].map(color_map)

        st.dataframe(note_stats[[
            "Status Icon", "status", "runs_seen", "example_description", "cluster_id"
        ]])

        with st.expander("📋 Note Appearance Matrix (hash × run)"):
            st.dataframe(result["note_history"].fillna(""))

    except Exception as e:
        st.warning(f"Unable to compute note changes: {e}")

    # --- Rule Drift Analysis ---
    st.subheader("📜 Rule Definition Drift Across Runs")
    try:
        rule_result = track_rule_changes_across_runs(run_paths)
        if rule_result is not None:
            rule_stats = rule_result["rule_stats"]
            rule_history = rule_result["rule_history"]

            st.markdown("### Rule Logic Change Summary")
            st.write({
                "Total Unique Rules (by logic)": len(rule_stats),
                "Stable (all runs)": int((rule_stats["runs_seen"] == len(run_paths)).sum()),
                "Changed/Intermittent": int((rule_stats["runs_seen"] < len(run_paths)).sum())
            })

            st.dataframe(rule_stats[[
                "rule_id", "description", "column", "type", "runs_seen"
            ]])

            with st.expander("📋 Rule Appearance Matrix (hash × run)"):
                st.dataframe(rule_history)
        else:
            st.info("No rules.json files found in selected runs.")
    except Exception as e:
        st.warning(f"Unable to compute rule changes: {e}")

import os
from datetime import datetime
import streamlit as st
from components.tabs import render_summary_tab, render_detailed_tab_with_interactions, render_historical_tab
from components.dq_card import dq_card
from utils.load_summary import load_violation_summary


def main():
    # Streamlit Config and Page Setup
    st.set_page_config(page_title="DQ Monitoring", layout="wide")
    st.title("Data Quality Dashboard")

    base_dir = "artifacts/runs"

    # Step 1: Select Domain
    try:
        domains = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    except FileNotFoundError:
        st.error("Could not find the `artifacts/runs` directory.")
        return

    if not domains:
        st.warning("No domains found in artifacts/runs.")
        return

    selected_domain = st.selectbox("Select Domain", domains)
    domain_path = os.path.join(base_dir, selected_domain)

    # Step 2: Select Run
    runs = sorted([r for r in os.listdir(domain_path) if os.path.isdir(os.path.join(domain_path, r))], reverse=True)
    if not runs:
        st.warning("No runs available for the selected domain.")
        return

    def format_run_name(name):
        try:
            dt = datetime.strptime(name, "%Y-%m-%d_%H-%M-%S")
            return dt.strftime("%b %d, %Y at %I:%M %p")
        except ValueError:
            return name

    run_display_names = [format_run_name(run) for run in runs]
    run_lookup = dict(zip(run_display_names, runs))
    selected_display_name = st.selectbox("Select Run", run_display_names)
    selected_run = run_lookup[selected_display_name]

    run_path = os.path.join(domain_path, selected_run)
    run_date = datetime.strptime(selected_run, "%Y-%m-%d_%H-%M-%S").date()

    # Step 3: Load and Display
    try:
        summary = load_violation_summary(run_path, run_date)
    except FileNotFoundError:
        st.error("Selected run folder or summary file not found.")
        return
    except Exception as e:
        st.error(f"An error occurred while loading the summary: {e}")
        return

    # Tabs for Views
    tabs = st.tabs(["Summary", "Detailed", "Historical"])
    with tabs[0]:
        render_summary_tab(summary, selected_display_name, dq_card)
    with tabs[1]:
        render_detailed_tab_with_interactions(run_path)
    with tabs[2]:
        render_historical_tab(domain_path)


if __name__ == "__main__":
    main()

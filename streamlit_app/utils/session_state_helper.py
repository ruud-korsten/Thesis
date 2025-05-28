import streamlit as st

def clear_all_filters(rules, notes):
    for rule in rules:
        st.session_state[f"checkbox_rule_{rule['id']}"] = False
    for note in notes:
        st.session_state[f"checkbox_note_{note['id']}"] = False
    st.session_state["checkbox_missing"] = False
    st.session_state["checkbox_duplicates"] = False

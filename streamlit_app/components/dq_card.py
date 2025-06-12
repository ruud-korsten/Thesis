import streamlit as st


def dq_card(column, title, data):
    """
    Renders a card for the Data Quality Dashboard.

    Args:
    - column: Streamlit column object to render the card in.
    - title (str): Title of the card.
    - data (dict): A dictionary with keys:
        - passed: Boolean indicating if the test passed.
        - summary: A short summary string.
        - details: A list of strings showing detailed information.
    """
    with column.container():
        st.subheader(title)
        if data["passed"]:
            st.success(data["summary"])
        else:
            st.error(data["summary"])
            with st.expander("Details"):
                for detail in data["details"]:
                    st.markdown(f"- {detail}")

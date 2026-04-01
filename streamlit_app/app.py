import streamlit as st

def main():
    st.set_page_config(page_title="CXR Model Focus Analysis", layout="wide")

    # Define your pages with custom titles and icons
    page_1 = st.Page("pages/triage.py", title="Model triage", icon=":material/edit:")
    page_2 = st.Page("pages/labeling.py", title="Labeling", icon=":material/label:")
    page_3 = st.Page("pages/model_metrics.py", title="Metrics analysis", icon=":material/analytics:")

    # Initialize the navigation
    pg = st.navigation([page_1, page_2, page_3])

    # Run the navigation
    pg.run()


if __name__ == "__main__":
    main()
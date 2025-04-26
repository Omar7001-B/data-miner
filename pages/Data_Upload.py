import streamlit as st
import pandas as pd

st.title("Data Upload")

# If a DataFrame is already in session_state, use it
if 'df' in st.session_state:
    df = st.session_state['df']
    st.success("Data already loaded! Showing preview:")
    rows_per_page = st.selectbox("Rows per page", [10, 25, 50, 100], index=0)
    total_rows = df.shape[0]
    total_pages = (total_rows - 1) // rows_per_page + 1
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    start = (page - 1) * rows_per_page
    end = start + rows_per_page
    st.dataframe(df.iloc[start:end])
else:
    uploaded_file = st.file_uploader(
        "Upload your data file (CSV, Excel, or JSON)",
        type=["csv", "xlsx", "xls", "json"]
    )
    if uploaded_file:
        file_type = uploaded_file.name.split(".")[-1].lower()
        try:
            if file_type in ["csv"]:
                df = pd.read_csv(uploaded_file)
            elif file_type in ["xlsx", "xls"]:
                df = pd.read_excel(uploaded_file)
            elif file_type == "json":
                df = pd.read_json(uploaded_file)
            st.session_state['df'] = df
            st.success(f"Loaded {uploaded_file.name}!")
            rows_per_page = st.selectbox("Rows per page", [10, 25, 50, 100], index=0)
            total_rows = df.shape[0]
            total_pages = (total_rows - 1) // rows_per_page + 1
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
            start = (page - 1) * rows_per_page
            end = start + rows_per_page
            st.dataframe(df.iloc[start:end])
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.info("Please upload a file to get started.") 
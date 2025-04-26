import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="DataMiner - Phase 1", layout="wide")
st.title("DataMiner: Interactive Data Mining")
st.header("Phase 1: Data Handling & Basic Preprocessing")

# File uploader
uploaded_file = st.file_uploader(
    "Upload your data file (CSV, Excel, or JSON)",
    type=["csv", "xlsx", "xls", "json"]
)

df = None
if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()
    try:
        if file_type in ["csv"]:
            df = pd.read_csv(uploaded_file)
        elif file_type in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
        elif file_type == "json":
            df = pd.read_json(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")

if df is not None:
    st.subheader("Data Preview")
    rows_per_page = st.selectbox("Rows per page", [10, 25, 50, 100], index=0)
    total_rows = df.shape[0]
    total_pages = (total_rows - 1) // rows_per_page + 1
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    start = (page - 1) * rows_per_page
    end = start + rows_per_page
    st.dataframe(df.iloc[start:end])

    st.subheader("Dataset Profiling")
    st.write("**Shape:**", df.shape)
    st.write("**Columns:**", list(df.columns))
    st.write("**Data Types:**")
    st.dataframe(df.dtypes.astype(str).reset_index().rename(columns={0: 'dtype', 'index': 'column'}))
    st.write("**Summary Statistics:**")
    st.dataframe(df.describe(include='all').T)

    st.write("**Correlation Matrix (numeric columns):**")
    corr = df.select_dtypes(include=[np.number]).corr()
    st.dataframe(corr)

    st.subheader("Modify Data Types")
    col1, col2 = st.columns([2, 3])
    with col1:
        column_to_change = st.selectbox("Select column to change type", df.columns)
    with col2:
        new_type = st.selectbox("New type", ["int", "float", "str", "datetime"], index=2)
    if st.button("Change Type"):
        try:
            if new_type == "int":
                df[column_to_change] = df[column_to_change].astype(int)
            elif new_type == "float":
                df[column_to_change] = df[column_to_change].astype(float)
            elif new_type == "str":
                df[column_to_change] = df[column_to_change].astype(str)
            elif new_type == "datetime":
                df[column_to_change] = pd.to_datetime(df[column_to_change], errors='coerce')
            st.success(f"Changed {column_to_change} to {new_type}")
        except Exception as e:
            st.error(f"Could not convert: {e}") 
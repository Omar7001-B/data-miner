import streamlit as st
import pandas as pd
import numpy as np

st.title("Summary Statistics")

df = st.session_state.get('df', None)
if df is not None:
    st.write("**Summary Statistics:**")
    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    if len(numeric_cols) > 0:
        st.markdown("**Numeric Columns**")
        st.dataframe(df[numeric_cols].describe().T)
    if len(categorical_cols) > 0:
        st.markdown("**Categorical Columns**")
        st.dataframe(df[categorical_cols].describe().T)

    with st.expander("What do the summary statistics fields mean?", expanded=False):
        st.markdown("""
        - **count**: Number of non-null entries in the column
        - **unique**: Number of unique values (categorical columns)
        - **top**: Most frequent value (categorical columns)
        - **freq**: Frequency of the most frequent value (categorical columns)
        - **mean**: Average value (numeric columns)
        - **std**: Standard deviation (numeric columns)
        - **min**: Minimum value (numeric columns)
        - **25%**: 25th percentile (numeric columns)
        - **50%**: Median/50th percentile (numeric columns)
        - **75%**: 75th percentile (numeric columns)
        - **max**: Maximum value (numeric columns)
        """)
else:
    st.warning("No data loaded. Please upload data on the Data Upload page.") 
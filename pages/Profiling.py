import streamlit as st
import pandas as pd

st.title("Data Profiling")

df = st.session_state.get('df', None)
if df is not None:
    st.write("**Shape:**", df.shape)
    st.write("**Columns:**", list(df.columns))
    st.write("**Data Types:**")
    st.dataframe(df.dtypes.astype(str).reset_index().rename(columns={0: 'dtype', 'index': 'column'}))
else:
    st.warning("No data loaded. Please upload data on the Data Upload page.") 
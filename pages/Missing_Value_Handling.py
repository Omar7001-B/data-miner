import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Missing Value Handling")

df = st.session_state.get('df', None)
if df is not None:
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    missing_summary = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing %': missing_percent
    })
    missing_summary = missing_summary[missing_summary['Missing Count'] > 0]

    tab1, tab2, tab3 = st.tabs(["Summary", "Visualization", "Handle Missing"])

    with tab1:
        st.subheader("Missing Values Summary Table")
        st.dataframe(missing_summary.style.background_gradient(cmap='YlOrRd', subset=['Missing %']))
        if not missing_summary.empty:
            st.subheader("Missing Values Bar Chart")
            st.bar_chart(missing_summary['Missing Count'])
        else:
            st.info("No missing values detected in your dataset!")

        # Before/After Table Comparison
        st.markdown("---")
        st.subheader("Preview: Before Handling (first 10 rows)")
        def highlight_missing(val):
            return 'background-color: #ffcccc' if pd.isnull(val) else ''
        st.dataframe(df.head(10).style.applymap(highlight_missing))
        if 'df_cleaned' in st.session_state:
            st.subheader("Preview: After Handling (first 10 rows)")
            st.dataframe(st.session_state['df_cleaned'].head(10))

    with tab2:
        if not missing_summary.empty:
            st.subheader("Missing Data Heatmap (Top 20 Columns, First 100 Rows)")
            top_missing_cols = missing_summary['Missing Count'].sort_values(ascending=False).head(20).index
            heatmap_data = df[top_missing_cols].isnull().iloc[:100]
            fig2, ax2 = plt.subplots(figsize=(max(10, len(top_missing_cols)), 4))
            sns.heatmap(
                heatmap_data,
                cbar=True,
                yticklabels=False,
                cmap="YlGnBu",
                ax=ax2,
                linewidths=0.5,
                linecolor='white'
            )
            ax2.set_title("Missing Data Heatmap (Top 20 Columns, First 100 Rows)")
            ax2.set_xlabel("Columns")
            ax2.set_ylabel("Rows (first 100)")
            plt.xticks(rotation=90, ha='center')
            st.pyplot(fig2)
            # Optional: Add missingno matrix visualization
            try:
                import missingno as msno
                st.subheader("Missing Data Matrix (missingno)")
                fig3 = msno.matrix(df, figsize=(10, 4))
                st.pyplot(fig3.figure)
            except ImportError:
                st.info("Install 'missingno' for an advanced missing data matrix visualization: pip install missingno")
        else:
            st.info("No missing values detected in your dataset!")

    with tab3:
        if not missing_summary.empty:
            st.subheader("Handle Missing Values")
            st.write("Select one or more columns to handle their missing values. Use the buttons below to select or clear all.")
            columns_with_missing = list(missing_summary.index)
            # Initialize session state for checkboxes
            if 'col_selection' not in st.session_state:
                st.session_state['col_selection'] = {col: False for col in columns_with_missing}
            # Buttons for select/clear all
            col1, col2 = st.columns([1,1])
            with col1:
                if st.button("Select All Columns"):
                    for col in columns_with_missing:
                        st.session_state['col_selection'][col] = True
            with col2:
                if st.button("Clear All Columns"):
                    for col in columns_with_missing:
                        st.session_state['col_selection'][col] = False
            # Show checkboxes for each column
            selected_cols = []
            for col in columns_with_missing:
                checked = st.checkbox(f"{col}", value=st.session_state['col_selection'][col], key=f"col_{col}")
                st.session_state['col_selection'][col] = checked
                if checked:
                    selected_cols.append(col)
            st.markdown(f"**Selected columns:** {', '.join(selected_cols) if selected_cols else 'None'}")
            method = st.selectbox("Imputation method", ["mean", "median", "mode", "drop rows", "drop columns"])
            threshold = None
            if method in ["drop rows", "drop columns"]:
                threshold = st.slider("Threshold (% of missing allowed before dropping)", 0, 100, 50)

            if st.button("Apply Handling"):
                if not selected_cols:
                    st.warning("Please select at least one column to handle.")
                else:
                    df_handled = df.copy()
                    if method in ["mean", "median", "mode"]:
                        for col in selected_cols:
                            if method == "mean" and pd.api.types.is_numeric_dtype(df_handled[col]):
                                df_handled[col] = df_handled[col].fillna(df_handled[col].mean())
                            elif method == "median" and pd.api.types.is_numeric_dtype(df_handled[col]):
                                df_handled[col] = df_handled[col].fillna(df_handled[col].median())
                            elif method == "mode":
                                mode_val = df_handled[col].mode()
                                if not mode_val.empty:
                                    df_handled[col] = df_handled[col].fillna(mode_val[0])
                    elif method == "drop rows":
                        df_handled = df_handled.dropna(axis=0, subset=selected_cols, thresh=int(len(selected_cols)*(1-threshold/100)))
                    elif method == "drop columns":
                        to_drop = [col for col in selected_cols if missing_percent[col] > threshold]
                        df_handled = df_handled.drop(columns=to_drop)
                    st.session_state['df_cleaned'] = df_handled
                    st.success("Missing value handling applied! See preview below.")

            if 'df_cleaned' in st.session_state:
                st.markdown("**Preview: Cleaned Data (first 10 rows)**")
                st.dataframe(st.session_state['df_cleaned'].head(10))
                st.download_button("Download Cleaned Data as CSV", st.session_state['df_cleaned'].to_csv(index=False), file_name="cleaned_data.csv")
                st.markdown("**Comparison: Before vs After (row counts)**")
                st.write(f"Original rows: {len(df)} | Cleaned rows: {len(st.session_state['df_cleaned'])}")
        else:
            st.info("No missing values detected in your dataset!")
else:
    st.warning("No data loaded. Please upload data on the Data Upload page.")
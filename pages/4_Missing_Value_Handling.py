import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    load_css, 
    create_footer, 
    show_file_required_warning, 
    display_dataset_info, 
    create_card,
    check_missing_values,
    plot_missing_heatmap
)

st.set_page_config(page_title="Missing Value Handling", page_icon="❓", layout="wide")
load_css()

st.title("Missing Value Handling ❓")
st.markdown("Identify, visualize, and handle missing values in your dataset with multiple imputation methods.")

df = st.session_state.get('df', None)
if df is not None:
    # Display dataset metrics
    display_dataset_info()
    
    # Calculate missing data using utility function
    missing_summary = check_missing_values(df)
    missing_summary = missing_summary[missing_summary['Missing Count'] > 0]
    
    # Create tabs for different aspects of missing value handling
    tab1, tab2, tab3 = st.tabs(["Summary", "Visualization", "Handle Missing"])

    with tab1:
        if not missing_summary.empty:
            st.subheader("Missing Values Overview")
            
            # Display missing values metrics
            metrics_cols = st.columns(4)
            with metrics_cols[0]:
                total_missing = missing_summary['Missing Count'].sum()
                total_cells = df.size
                st.metric("Total Missing Values", f"{total_missing:,}")
            
            with metrics_cols[1]:
                missing_percent_total = (total_missing / total_cells) * 100
                st.metric("Dataset Completion", f"{100 - missing_percent_total:.2f}%")
            
            with metrics_cols[2]:
                cols_with_missing = len(missing_summary)
                st.metric("Columns with Missing", f"{cols_with_missing}/{df.shape[1]}")
            
            with metrics_cols[3]:
                rows_with_missing = df.isnull().any(axis=1).sum()
                st.metric("Rows with Missing", f"{rows_with_missing:,} ({rows_with_missing/len(df)*100:.1f}%)")
            
            # Show summary table
            st.subheader("Missing Values by Column")
            st.dataframe(
                missing_summary.style.background_gradient(cmap='YlOrRd', subset=['Missing %']),
                use_container_width=True
            )
            
            # Missing Values Bar Chart
            st.subheader("Missing Values Bar Chart")
            fig, ax = plt.subplots(figsize=(10, max(5, len(missing_summary)*0.3)))
            
            # Reset index to make column names accessible as a column
            plot_df = missing_summary.reset_index().rename(columns={'index': 'Column'})
            bars = plot_df.sort_values('Missing Count', ascending=False).plot.barh(
                x='Column',
                y='Missing Count',
                ax=ax,
                color='#FF4B4B'
            )
            
            # Add percentage labels
            for i, p in enumerate(ax.patches):
                percentage = plot_df['Missing %'].sort_values(ascending=False).iloc[i]
                ax.annotate(f"{percentage:.1f}%",
                           (p.get_width() + 5, p.get_y() + p.get_height()/2.),
                           ha='left', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            create_card(
                "No Missing Values",
                "Your dataset doesn't have any missing values. Great news! Your data is already complete.",
                icon="✅"
            )

        # Before/After Table Comparison
        if 'df_cleaned' in st.session_state:
            st.markdown("---")
            st.subheader("Before vs After Comparison")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Before Handling (first 10 rows)**")
                def highlight_missing(val):
                    return 'background-color: #ffcccc' if pd.isnull(val) else ''
                st.dataframe(df.head(10).style.applymap(highlight_missing), use_container_width=True)
            
            with col2:
                st.markdown("**After Handling (first 10 rows)**")
                st.dataframe(st.session_state['df_cleaned'].head(10), use_container_width=True)
            
            # Show statistics on what changed
            original_missing = df.isnull().sum().sum()
            cleaned_missing = st.session_state['df_cleaned'].isnull().sum().sum()
            values_filled = original_missing - cleaned_missing
            
            metrics_cols = st.columns(3)
            with metrics_cols[0]:
                st.metric("Original Missing", f"{original_missing:,}")
            with metrics_cols[1]:
                st.metric("Values Filled", f"{values_filled:,}")
            with metrics_cols[2]:
                st.metric("Remaining Missing", f"{cleaned_missing:,}")

    with tab2:
        if not missing_summary.empty:
            st.subheader("Missing Data Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Heatmap of missing values using utility function
                st.markdown("#### Missing Data Heatmap")
                st.markdown("*Shows patterns of missingness (first 100 rows)*")
                
                # Use utility function to create heatmap
                heatmap_fig = plot_missing_heatmap(df, top_n=20, max_rows=100)
                st.pyplot(heatmap_fig)
            
            with col2:
                # Correlation of missingness
                st.markdown("#### Correlation of Missingness")
                st.markdown("*Shows if missing values in one column relate to another column*")
                
                top_missing_cols = missing_summary.index[:20]
                if len(top_missing_cols) > 1:
                    # Create binary missingness indicators
                    miss_corr_df = pd.DataFrame()
                    for col in top_missing_cols:
                        miss_corr_df[f"{col}_missing"] = df[col].isnull().astype(int)
                    
                    # Calculate correlation
                    miss_corr = miss_corr_df.corr()
                    
                    # Visualize correlation
                    fig3, ax3 = plt.subplots(figsize=(10, 8))
                    mask = np.triu(np.ones_like(miss_corr, dtype=bool))
                    sns.heatmap(
                        miss_corr, 
                        mask=mask,
                        cmap="coolwarm", 
                        vmin=-1, 
                        vmax=1, 
                        center=0,
                        square=True, 
                        linewidths=.5, 
                        annot=True, 
                        fmt=".2f",
                        ax=ax3
                    )
                    ax3.set_title("Correlation of Missingness Between Columns")
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    plt.tight_layout()
                    st.pyplot(fig3)
                else:
                    st.info("Need at least two columns with missing values for correlation analysis.")
            
            # Missingness by data type
            st.markdown("#### Missing Values by Data Type")
            
            # Group by data type
            dtype_missing = {}
            for dtype in df.dtypes.astype(str).unique():
                cols = df.select_dtypes(include=[dtype]).columns
                missing = df[cols].isnull().sum().sum()
                dtype_missing[dtype] = missing
            
            # Create visualization
            fig4, ax4 = plt.subplots(figsize=(10, 5))
            bars = ax4.bar(dtype_missing.keys(), dtype_missing.values(), color=['#FF4B4B', '#0068C9', '#83C9FF', '#29B09D'])
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:,}', ha='center', va='bottom')
            
            ax4.set_title("Missing Values by Data Type")
            ax4.set_xlabel("Data Type")
            ax4.set_ylabel("Count of Missing Values")
            plt.tight_layout()
            st.pyplot(fig4)
            
            # Optional: Add missingno matrix visualization
            try:
                import missingno as msno
                st.markdown("#### Missing Data Pattern (missingno matrix)")
                
                # Use Matplotlib figure
                with st.spinner("Generating missingno matrix..."):
                    plt.figure(figsize=(12, 6))
                    msno.matrix(df, figsize=(12, 6))
                    st.pyplot(plt.gcf())
            except ImportError:
                st.info("Install 'missingno' for an advanced missing data matrix visualization: pip install missingno")
                
        else:
            create_card(
                "No Missing Values to Visualize",
                "Your dataset is complete without any missing values. No visualizations needed!",
                icon="📊"
            )

    with tab3:
        if not missing_summary.empty:
            st.subheader("Handle Missing Values")
            
            st.markdown("""
            Select one or more columns to handle missing values using various imputation methods.
            """)
            
            # Column selection with enhanced UI
            columns_with_missing = list(missing_summary.index)
            
            # Create two columns
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Initialize session state for checkboxes if not exists
                if 'col_selection' not in st.session_state:
                    st.session_state['col_selection'] = {col: False for col in columns_with_missing}
                # --- Sync col_selection with current columns_with_missing ---
                for col in columns_with_missing:
                    if col not in st.session_state['col_selection']:
                        st.session_state['col_selection'][col] = False
                for col in list(st.session_state['col_selection'].keys()):
                    if col not in columns_with_missing:
                        del st.session_state['col_selection'][col]
                # --- End sync ---

                # Show column type in selection UI
                col_types = {col: str(df[col].dtype) for col in columns_with_missing}

                # Group columns by their missing percentage for better organization
                high_missing = [col for col in columns_with_missing if missing_summary.loc[col, 'Missing %'] > 50]
                medium_missing = [col for col in columns_with_missing if 10 <= missing_summary.loc[col, 'Missing %'] <= 50]
                low_missing = [col for col in columns_with_missing if missing_summary.loc[col, 'Missing %'] < 10]

                def checkbox_label(col):
                    return f"{col} [{col_types[col]}] ({missing_summary.loc[col, 'Missing Count']:,} missing, {missing_summary.loc[col, 'Missing %']:.1f}%)"

                # Create expandable sections for each group
                for group, group_name in zip([high_missing, medium_missing, low_missing],
                                             ["Columns with High Missing Values (>50%)",
                                              "Columns with Medium Missing Values (10-50%)",
                                              "Columns with Low Missing Values (<10%)"]):
                    if group:
                        with st.expander(group_name, expanded=True):
                            for col in group:
                                checked = st.checkbox(
                                    checkbox_label(col),
                                    value=st.session_state['col_selection'][col],
                                    key=f"col_{col}"
                                )
                                st.session_state['col_selection'][col] = checked

            # --- Ensure selected_cols is always defined before use ---
            selected_cols = [col for col in columns_with_missing if st.session_state['col_selection'][col]]
            st.markdown(f"**Selected columns:** {', '.join(selected_cols) if selected_cols else 'None'}")

            with col2:
                st.markdown("#### Imputation Method")
                if selected_cols:
                    # Determine types of selected columns
                    selected_types = {col: str(df[col].dtype) for col in selected_cols}
                    all_numeric = all(pd.api.types.is_numeric_dtype(df[col]) for col in selected_cols)
                    all_categorical = all(pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]) for col in selected_cols)

                    # Show only valid methods for the selection
                    if all_numeric:
                        method = st.radio(
                            "Choose method for numeric columns:",
                            [
                                ("mean", "Fill with mean (average) value"),
                                ("median", "Fill with median value"),
                                ("mode", "Fill with most frequent value (mode)"),
                                ("fill value", "Fill with a custom value"),
                                ("drop rows", "Drop rows with missing values"),
                                ("drop columns", "Drop columns with high missing values")
                            ],
                            format_func=lambda x: x[1],
                            key="impute_method_numeric"
                        )[0]
                    elif all_categorical:
                        method = st.radio(
                            "Choose method for categorical columns:",
                            [
                                ("mode", "Fill with most frequent value (mode)"),
                                ("fill value", "Fill with a custom value"),
                                ("drop rows", "Drop rows with missing values"),
                                ("drop columns", "Drop columns with high missing values")
                            ],
                            format_func=lambda x: x[1],
                            key="impute_method_categorical"
                        )[0]
                    else:
                        method = st.radio(
                            "Choose method (mixed types):",
                            [
                                ("mode", "Fill with most frequent value (mode)"),
                                ("fill value", "Fill with a custom value"),
                                ("drop rows", "Drop rows with missing values"),
                                ("drop columns", "Drop columns with high missing values")
                            ],
                            format_func=lambda x: x[1],
                            key="impute_method_mixed"
                        )[0]

                    # Additional options based on method
                    fill_value = None
                    if method == "fill value":
                        fill_value = st.text_input("Value to fill with", "0")

                    threshold = None
                    if method in ["drop rows", "drop columns"]:
                        threshold = st.slider(
                            "Threshold (% of missing allowed)",
                            0, 100, 50,
                            help="For 'drop rows': Remove rows with more missing % than threshold\nFor 'drop columns': Remove columns with more missing % than threshold"
                        )
                else:
                    method = None

            # --- Imputation method selection ---
            selected_cols = [col for col in columns_with_missing if st.session_state['col_selection'][col]]
            st.markdown(f"**Selected columns:** {', '.join(selected_cols) if selected_cols else 'None'}")

            # --- Apply button and imputation logic ---
            if st.button("Apply Handling", disabled=not selected_cols or not method):
                progress_bar = st.progress(0)
                status_text = st.empty()
                df_handled = df.copy()
                for i, col in enumerate(selected_cols):
                    progress = int((i / len(selected_cols)) * 100)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing column: {col}...")
                    col_type = str(df[col].dtype)
                    try:
                        if method == "mean" and pd.api.types.is_numeric_dtype(df_handled[col]):
                            df_handled[col] = df_handled[col].fillna(df_handled[col].mean())
                        elif method == "median" and pd.api.types.is_numeric_dtype(df_handled[col]):
                            df_handled[col] = df_handled[col].fillna(df_handled[col].median())
                        elif method == "mode":
                            mode_val = df_handled[col].mode()
                            if not mode_val.empty:
                                df_handled[col] = df_handled[col].fillna(mode_val[0])
                        elif method == "fill value":
                            if pd.api.types.is_numeric_dtype(df_handled[col]):
                                try:
                                    df_handled[col] = df_handled[col].fillna(float(fill_value))
                                except Exception:
                                    df_handled[col] = df_handled[col].fillna(str(fill_value))
                            elif pd.api.types.is_datetime64_dtype(df_handled[col]):
                                try:
                                    df_handled[col] = df_handled[col].fillna(pd.to_datetime(fill_value))
                                except Exception:
                                    df_handled[col] = df_handled[col].fillna(str(fill_value))
                            else:
                                df_handled[col] = df_handled[col].fillna(str(fill_value))
                    except Exception as e:
                        st.warning(f"Could not impute column {col}: {e}")
                # For row or column dropping, apply at the end
                if method == "drop rows":
                    orig_rows = len(df_handled)
                    df_handled = df_handled.dropna(
                        axis=0,
                        subset=selected_cols,
                        thresh=int(len(selected_cols)*(1-threshold/100))
                    )
                    rows_dropped = orig_rows - len(df_handled)
                    status_text.text(f"Dropped {rows_dropped} rows with high missing values.")
                elif method == "drop columns":
                    to_drop = [col for col in selected_cols if missing_summary.loc[col, 'Missing %'] > threshold]
                    if to_drop:
                        df_handled = df_handled.drop(columns=to_drop)
                        status_text.text(f"Dropped {len(to_drop)} columns with high missing values: {', '.join(to_drop)}")
                    else:
                        status_text.text("No columns meet the threshold criteria for dropping.")
                progress_bar.progress(100)
                st.session_state['df_cleaned'] = df_handled
                st.success("✅ Missing value handling applied successfully!")
                st.download_button(
                    "📥 Download Cleaned Data as CSV",
                    df_handled.to_csv(index=False),
                    file_name="cleaned_data.csv",
                    mime="text/csv"
                )
                if st.button("📥 Replace Original Data with Cleaned Version"):
                    st.session_state['df'] = df_handled.copy()
                    st.success("Original data replaced with cleaned version!")
                    st.rerun()
            
            # Display result if available
            if 'df_cleaned' in st.session_state:
                with st.expander("Preview Cleaned Data", expanded=True):
                    st.dataframe(st.session_state['df_cleaned'].head(10), use_container_width=True)
                    
                    # Show statistics 
                    original_nulls = df.isnull().sum().sum()
                    cleaned_nulls = st.session_state['df_cleaned'].isnull().sum().sum()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Missing Values", f"{original_nulls:,}")
                    with col2:
                        st.metric("Current Missing Values", f"{cleaned_nulls:,}")
                    with col3:
                        pct_reduction = 0 if original_nulls == 0 else (original_nulls - cleaned_nulls) / original_nulls * 100
                        st.metric("Reduction in Missing Values", f"{pct_reduction:.1f}%")
        else:
            create_card(
                "No Missing Values to Handle",
                """
                Your dataset doesn't have any missing values to handle. You can proceed to other analyses.
                
                If you want to experiment with missing value handling, try creating some missing values manually:
                ```python
                # Example of how to create missing values
                import numpy as np
                df['column_name'] = df['column_name'].mask(np.random.random(len(df)) < 0.1) # Adds 10% missing values
                ```
                """,
                icon="🎯"
            )
else:
    show_file_required_warning()

# Footer
create_footer()
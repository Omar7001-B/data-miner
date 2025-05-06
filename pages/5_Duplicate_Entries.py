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
    create_card
)

st.set_page_config(page_title="Duplicate Entries", page_icon="🔍", layout="wide")
load_css()

st.title("Duplicate Entries Removal 🔍")
st.markdown("Identify, visualize, and remove duplicate entries in your dataset to ensure data quality.")

df = st.session_state.get('df', None)
if df is not None:
    # Display dataset metrics
    display_dataset_info()
    
    # Check for duplicate rows
    if 'duplicate_subset' not in st.session_state:
        st.session_state['duplicate_subset'] = None
    
    # Create tabs for different aspects of duplicate handling
    tab1, tab2, tab3 = st.tabs(["Summary", "Visualization", "Remove Duplicates"])

    with tab1:
        st.subheader("Duplicate Entries Overview")
        
        # Check for duplicates using all columns
        duplicates_all = df.duplicated().sum()
        total_rows = len(df)
        
        # Display duplicate metrics
        metrics_cols = st.columns(4)
        with metrics_cols[0]:
            st.metric("Total Rows", f"{total_rows:,}")
        
        with metrics_cols[1]:
            st.metric("Duplicate Rows", f"{duplicates_all:,}")
        
        with metrics_cols[2]:
            duplicate_percent = (duplicates_all / total_rows) * 100
            st.metric("Duplicate Percentage", f"{duplicate_percent:.2f}%")
        
        with metrics_cols[3]:
            unique_rows = total_rows - duplicates_all
            st.metric("Unique Rows", f"{unique_rows:,}")
        
        if duplicates_all > 0:
            st.success(f"Found {duplicates_all:,} duplicate rows in the dataset.")
            
            # Show sample of duplicate rows
            st.subheader("Sample of Duplicate Entries")
            duplicate_mask = df.duplicated(keep=False)
            duplicate_samples = df[duplicate_mask].head(10)
            
            st.dataframe(duplicate_samples, use_container_width=True)
        else:
            create_card(
                "No Duplicate Entries",
                "Your dataset doesn't have any duplicate rows considering all columns. Great news!",
                icon="✅"
            )
        
        # Feature for checking duplicates based on specific columns
        st.subheader("Check Duplicates by Specific Columns")
        st.markdown("Select columns to identify duplicates based only on those columns:")
        
        # Column selector
        selected_columns = st.multiselect(
            "Select columns to check for duplicates",
            options=df.columns.tolist(),
            default=[]
        )
        
        if selected_columns:
            st.session_state['duplicate_subset'] = selected_columns
            duplicates_subset = df.duplicated(subset=selected_columns, keep=False).sum()
            unique_subset = total_rows - df.duplicated(subset=selected_columns).sum()
            
            subset_cols = st.columns(3)
            with subset_cols[0]:
                st.metric("Rows with Duplicates", f"{duplicates_subset:,}")
            with subset_cols[1]:
                duplicate_subset_percent = (duplicates_subset / total_rows) * 100
                st.metric("Duplicate Percentage", f"{duplicate_subset_percent:.2f}%")
            with subset_cols[2]:
                st.metric("Unique Entries", f"{unique_subset:,}")
            
            if duplicates_subset > 0:
                st.markdown("### Sample of Duplicates Based on Selected Columns")
                subset_duplicates = df[df.duplicated(subset=selected_columns, keep=False)]
                st.dataframe(subset_duplicates.sort_values(by=selected_columns).head(10), use_container_width=True)
            else:
                st.success("No duplicates found with the selected columns.")

    with tab2:
        st.subheader("Duplicate Data Visualizations")
        
        # Check if we have duplicates to visualize
        has_all_duplicates = df.duplicated(keep=False).sum() > 0
        has_subset_duplicates = st.session_state['duplicate_subset'] is not None and df.duplicated(subset=st.session_state['duplicate_subset'], keep=False).sum() > 0
        
        if has_all_duplicates or has_subset_duplicates:
            # Decide which duplicates to visualize
            if has_subset_duplicates:
                duplicate_mask = df.duplicated(subset=st.session_state['duplicate_subset'], keep=False)
                duplicate_title = f"Duplicates based on columns: {', '.join(st.session_state['duplicate_subset'])}"
            else:
                duplicate_mask = df.duplicated(keep=False)
                duplicate_title = "Duplicates based on all columns"
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Duplicate count by categorical variable
                st.markdown("#### Duplicates by Categorical Variable")
                
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                if categorical_cols:
                    selected_cat = st.selectbox(
                        "Select categorical column for analysis",
                        options=categorical_cols
                    )
                    
                    # Count duplicates by category
                    dup_by_cat = df[duplicate_mask][selected_cat].value_counts().reset_index()
                    dup_by_cat.columns = [selected_cat, 'Duplicate Count']
                    
                    # Display a horizontal bar chart for top categories
                    fig, ax = plt.subplots(figsize=(10, 6))
                    top_n = min(10, len(dup_by_cat))
                    sns.barplot(
                        data=dup_by_cat.head(top_n),
                        y=selected_cat,
                        x='Duplicate Count',
                        ax=ax,
                        color='#FF4B4B'
                    )
                    ax.set_title(f'Top {top_n} {selected_cat} with Duplicates')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show the data in a table
                    st.dataframe(dup_by_cat.head(10), use_container_width=True)
                else:
                    st.info("No categorical columns found in the dataset.")
            
            with col2:
                # Duplicate distribution visualization
                st.markdown("#### Duplicate Distribution")
                
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    selected_num = st.selectbox(
                        "Select numeric column for distribution",
                        options=numeric_cols
                    )
                    
                    # Create histogram comparing distribution of duplicates vs non-duplicates
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plot both distributions
                    sns.histplot(
                        data=df[~duplicate_mask], 
                        x=selected_num, 
                        alpha=0.5, 
                        color="blue", 
                        label="Unique Entries", 
                        ax=ax
                    )
                    sns.histplot(
                        data=df[duplicate_mask], 
                        x=selected_num, 
                        alpha=0.5, 
                        color="red", 
                        label="Duplicate Entries", 
                        ax=ax
                    )
                    ax.set_title(f'Distribution of {selected_num} - Unique vs Duplicate Entries')
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show descriptive statistics
                    stats_col1, stats_col2 = st.columns(2)
                    with stats_col1:
                        st.markdown("**Unique Entries Stats**")
                        st.dataframe(df[~duplicate_mask][selected_num].describe())
                    with stats_col2:
                        st.markdown("**Duplicate Entries Stats**")
                        st.dataframe(df[duplicate_mask][selected_num].describe())
                else:
                    st.info("No numeric columns found in the dataset.")
            
            # Additional visualization: Duplicate counts by group
            if has_subset_duplicates and len(st.session_state['duplicate_subset']) > 0:
                st.markdown("#### Duplicate Counts by Group")
                
                # Count occurrences of each group
                duplicate_counts = df.groupby(st.session_state['duplicate_subset']).size().reset_index(name='count')
                duplicate_counts = duplicate_counts[duplicate_counts['count'] > 1].sort_values('count', ascending=False)
                
                if not duplicate_counts.empty:
                    st.markdown(f"**Groups with Multiple Occurrences (Top 10)**")
                    st.dataframe(duplicate_counts.head(10), use_container_width=True)
                    
                    # Visualize distribution of duplicate counts
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(
                        data=duplicate_counts,
                        x='count',
                        bins=min(20, len(duplicate_counts)),
                        kde=True,
                        color='purple',
                        ax=ax
                    )
                    ax.set_title('Distribution of Duplicate Group Sizes')
                    ax.set_xlabel('Number of Occurrences per Group')
                    ax.set_ylabel('Frequency')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("No duplicate groups found with the selected columns.")
        else:
            create_card(
                "No Duplicates to Visualize",
                "There are no duplicate entries in your dataset to visualize.",
                icon="📊"
            )

    with tab3:
        st.subheader("Remove Duplicate Entries")
        
        # Choose how to find duplicates
        duplicate_type = st.radio(
            "How to identify duplicates?",
            ["All Columns", "Specific Columns"],
            help="'All Columns': Rows with identical values across all columns\n'Specific Columns': Rows with identical values in selected columns"
        )
        
        if duplicate_type == "Specific Columns":
            # Column selector for duplicates
            dedup_columns = st.multiselect(
                "Select columns to check for duplicates",
                options=df.columns.tolist(),
                default=st.session_state.get('duplicate_subset', [])
            )
            
            if not dedup_columns:
                st.warning("⚠️ Please select at least one column to identify duplicates.")
                has_duplicates = False
            else:
                duplicate_count = df.duplicated(subset=dedup_columns, keep=False).sum()
                has_duplicates = duplicate_count > 0
                
                if has_duplicates:
                    st.success(f"Found {duplicate_count:,} rows with duplicated values in the selected columns.")
                else:
                    st.info("No duplicates found with the selected columns.")
        else:  # All Columns
            duplicate_count = df.duplicated().sum()
            has_duplicates = duplicate_count > 0
            
            if has_duplicates:
                st.success(f"Found {duplicate_count:,} duplicate rows in the dataset.")
            else:
                st.info("No duplicate rows found in the dataset.")
        
        if has_duplicates:
            st.markdown("### Duplicate Removal Options")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                keep_option = st.radio(
                    "Which duplicates to keep?",
                    ["first", "last", "none"],
                    help="'first': Keep first occurrence\n'last': Keep last occurrence\n'none': Drop all duplicates"
                )
            
            with col2:
                st.markdown("""
                **Duplicate Removal Strategies:**
                - **Keep first**: Keep the first occurrence of duplicated rows, drop subsequent duplicates
                - **Keep last**: Keep the last occurrence of duplicated rows, drop previous duplicates
                - **Keep none**: Drop all duplicate rows (keeps no duplicates)
                """)
            
            # Apply duplicate removal
            if st.button("Remove Duplicates", key="remove_duplicates_btn"):
                with st.spinner("Removing duplicate entries..."):
                    # Make a copy of the original DataFrame
                    df_dedup = df.copy()
                    
                    # Remove duplicates based on options
                    if duplicate_type == "Specific Columns":
                        before_count = len(df_dedup)
                        df_dedup = df_dedup.drop_duplicates(subset=dedup_columns, keep=keep_option)
                        removed_count = before_count - len(df_dedup)
                    else:  # All Columns
                        before_count = len(df_dedup)
                        df_dedup = df_dedup.drop_duplicates(keep=keep_option)
                        removed_count = before_count - len(df_dedup)
                    
                    # Store the result
                    st.session_state['df_dedup'] = df_dedup
                    
                    st.success(f"✅ Successfully removed {removed_count:,} duplicate rows!")
                    
                    # Show download button
                    st.download_button(
                        "📥 Download Deduplicated Data as CSV", 
                        df_dedup.to_csv(index=False), 
                        file_name="deduplicated_data.csv",
                        mime="text/csv"
                    )
                    
                    # Offer option to replace original dataframe
                    if st.button("📥 Replace Original Data with Deduplicated Version"):
                        st.session_state['df'] = df_dedup.copy()
                        st.success("Original data replaced with deduplicated version!")
                        st.rerun()
            
            # Show result if available
            if 'df_dedup' in st.session_state:
                with st.expander("Preview Deduplicated Data", expanded=True):
                    st.dataframe(st.session_state['df_dedup'].head(10), use_container_width=True)
                    
                    # Compare with original
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Row Count", f"{len(df):,}")
                    with col2:
                        st.metric("Deduplicated Row Count", f"{len(st.session_state['df_dedup']):,}")
                    with col3:
                        removed = len(df) - len(st.session_state['df_dedup'])
                        removed_pct = (removed / len(df)) * 100 if len(df) > 0 else 0
                        st.metric("Rows Removed", f"{removed:,} ({removed_pct:.1f}%)")
        else:
            if duplicate_type == "All Columns":
                create_card(
                    "No Duplicates to Remove",
                    "Your dataset doesn't have any duplicate rows. No action needed!",
                    icon="✅"
                )
else:
    show_file_required_warning()

# Footer
create_footer() 
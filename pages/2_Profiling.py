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
    get_numeric_columns,
    get_categorical_columns,
    plot_distribution
)

st.set_page_config(page_title="Data Profiling", page_icon="🔍", layout="wide")
load_css()

st.title("Data Profiling 🔍")
st.markdown("Explore your dataset structure and understand its characteristics.")

df = st.session_state.get('df', None)
if df is not None:
    # Display data metrics at the top
    display_dataset_info()
    
    # Create tabs for different profiling aspects
    tab1, tab2, tab3 = st.tabs(["Overview", "Column Details", "Type Distribution"])
    
    with tab1:
        st.subheader("Dataset Overview")
        
        # Dataset shape and memory usage
        col1, col2 = st.columns(2)
        with col1:
            st.write("**📏 Shape:**", df.shape)
        with col2:
            memory_usage = df.memory_usage(deep=True).sum()
            st.write("**💾 Memory Usage:**", f"{memory_usage / 1024**2:.2f} MB")
        
        # Column list with types
        st.write("**📋 Columns:**")
        dtypes_df = df.dtypes.reset_index()
        dtypes_df.columns = ["Column Name", "Data Type"]
        dtypes_df["Data Type"] = dtypes_df["Data Type"].astype(str)
        
        # Add null counts and percent
        dtypes_df["Null Count"] = df.isnull().sum().values
        dtypes_df["Null %"] = dtypes_df["Null Count"] / len(df) * 100
        dtypes_df["Null %"] = dtypes_df["Null %"].round(2)
        
        # Add unique counts and percent
        dtypes_df["Unique Values"] = df.nunique().values
        dtypes_df["Unique %"] = (dtypes_df["Unique Values"] / len(df) * 100).round(2)
        
        st.dataframe(dtypes_df, use_container_width=True)
    
    with tab2:
        st.subheader("Column Details")
        
        # Select a column to explore
        col_to_explore = st.selectbox("Select a column to explore", df.columns)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**📊 Statistics for {col_to_explore}**")
            
            # Get column stats based on type
            col_data = df[col_to_explore]
            
            if pd.api.types.is_numeric_dtype(col_data):
                stats = pd.DataFrame({
                    "Statistic": ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max", "Unique Values", "Null Count", "Null %"],
                    "Value": [
                        len(col_data),
                        col_data.mean(),
                        col_data.std(),
                        col_data.min(),
                        col_data.quantile(0.25),
                        col_data.median(),
                        col_data.quantile(0.75),
                        col_data.max(),
                        col_data.nunique(),
                        col_data.isnull().sum(),
                        col_data.isnull().sum() / len(col_data) * 100
                    ]
                })
            else:
                # For non-numeric columns
                stats = pd.DataFrame({
                    "Statistic": ["Count", "Unique Values", "Top Value", "Top Frequency", "Null Count", "Null %"],
                    "Value": [
                        len(col_data),
                        col_data.nunique(),
                        col_data.mode()[0] if not col_data.mode().empty else None,
                        col_data.value_counts().iloc[0] if not col_data.value_counts().empty else 0,
                        col_data.isnull().sum(),
                        col_data.isnull().sum() / len(col_data) * 100
                    ]
                })
            
            st.dataframe(stats, use_container_width=True)
            
        with col2:
            st.write(f"**📈 Visualization for {col_to_explore}**")
            
            # Use the utility function to plot distribution
            fig = plot_distribution(df, col_to_explore)
            st.pyplot(fig)
            
            # Show top and bottom values
            if not col_data.empty:
                st.write("**Sample Values:**")
                sample_col1, sample_col2 = st.columns(2)
                with sample_col1:
                    st.write("First 5 Values:")
                    st.write(col_data.head())
                with sample_col2:
                    st.write("Last 5 Values:")
                    st.write(col_data.tail())
    
    with tab3:
        st.subheader("Data Type Distribution")
        
        # Generate pie chart of data types
        type_counts = df.dtypes.astype(str).value_counts()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Data types pie chart
        ax1.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Data Type Distribution')
        ax1.axis('equal')
        
        # Missing values by data type
        missing_by_type = {}
        for dtype in type_counts.index:
            cols = df.select_dtypes(include=[dtype]).columns
            missing = df[cols].isnull().sum().sum()
            missing_by_type[dtype] = missing
        
        missing_series = pd.Series(missing_by_type)
        if missing_series.sum() > 0:
            ax2.bar(missing_series.index, missing_series.values)
            ax2.set_title('Missing Values by Data Type')
            plt.xticks(rotation=45, ha='right')
            ax2.set_ylabel('Count of Missing Values')
        else:
            ax2.text(0.5, 0.5, 'No Missing Values', horizontalalignment='center', verticalalignment='center')
            ax2.set_title('Missing Values by Data Type')
            ax2.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Summary of data types
        st.write("**Type Summary:**")
        st.write(f"- Numeric Columns: {len(get_numeric_columns(df))}")
        st.write(f"- Categorical/Object Columns: {len(get_categorical_columns(df))}")
        st.write(f"- DateTime Columns: {len(df.select_dtypes(include=['datetime']).columns)}")
        st.write(f"- Boolean Columns: {len(df.select_dtypes(include=['bool']).columns)}")

else:
    show_file_required_warning()

# Footer
create_footer() 
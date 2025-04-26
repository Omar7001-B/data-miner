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
    generate_summary_stats,
    plot_correlation_heatmap
)

st.set_page_config(page_title="Summary Statistics", page_icon="📊", layout="wide")
load_css()

st.title("Summary Statistics 📊")
st.markdown("Get comprehensive statistics and insights about your dataset's values.")

df = st.session_state.get('df', None)
if df is not None:
    # Display dataset metrics
    display_dataset_info()
    
    # Create tabs for different types of statistics
    tab1, tab2, tab3 = st.tabs(["Numeric Stats", "Categorical Stats", "Correlation Analysis"])
    
    with tab1:
        numeric_cols = get_numeric_columns(df)
        if len(numeric_cols) > 0:
            st.subheader("Numeric Column Statistics")
            
            # Column selector
            cols_to_analyze = st.multiselect(
                "Select columns to analyze:",
                options=numeric_cols,
                default=list(numeric_cols[:min(5, len(numeric_cols))])
            )
            
            if cols_to_analyze:
                # Use the utility function for summary stats
                stats_df = generate_summary_stats(df, cols_to_analyze)
                st.dataframe(stats_df.round(2), use_container_width=True)
                
                # Visualizations
                st.subheader("Distribution Visualization")
                
                # Box plot
                fig, ax = plt.subplots(figsize=(12, 5))
                sns.boxplot(data=df[cols_to_analyze], ax=ax)
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
                
                # Plot histograms for selected columns
                if len(cols_to_analyze) > 0:
                    st.subheader("Histograms")
                    num_cols = min(len(cols_to_analyze), 4)  # Up to 4 columns per row
                    num_rows = (len(cols_to_analyze) - 1) // num_cols + 1
                    
                    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 3 * num_rows))
                    if num_rows == 1 and num_cols == 1:
                        axes = np.array([axes])
                    axes = axes.flatten()
                    
                    for i, col in enumerate(cols_to_analyze):
                        if i < len(axes):
                            sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
                            axes[i].set_title(f"Distribution of {col}")
                            axes[i].tick_params(labelrotation=45)
                    
                    # Hide unused subplots
                    for j in range(len(cols_to_analyze), len(axes)):
                        axes[j].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info("Please select at least one numeric column to analyze.")
        else:
            st.info("No numeric columns found in the dataset.")
    
    with tab2:
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            st.subheader("Categorical Column Statistics")
            
            # Column selector
            selected_cat_col = st.selectbox("Select a categorical column:", categorical_cols)
            
            if selected_cat_col:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Enhanced categorical stats with percentages
                    value_counts = df[selected_cat_col].value_counts().reset_index()
                    value_counts.columns = [selected_cat_col, 'Count']
                    value_counts['Percentage'] = (value_counts['Count'] / len(df)) * 100
                    
                    # Limit to top 20 for display
                    if len(value_counts) > 20:
                        st.write(f"Showing top 20 of {len(value_counts)} categories")
                        value_counts = value_counts.head(20)
                    
                    st.dataframe(value_counts, use_container_width=True)
                
                with col2:
                    # Summary statistics of the categorical column
                    cat_stats = pd.DataFrame({
                        'Statistic': [
                            'Total Count', 
                            'Unique Values', 
                            'Most Common', 
                            'Most Common Count',
                            'Most Common %',
                            'Missing Values',
                            'Missing %'
                        ],
                        'Value': [
                            len(df[selected_cat_col]),
                            df[selected_cat_col].nunique(),
                            df[selected_cat_col].value_counts().index[0] if not df[selected_cat_col].value_counts().empty else None,
                            df[selected_cat_col].value_counts().iloc[0] if not df[selected_cat_col].value_counts().empty else 0,
                            df[selected_cat_col].value_counts().iloc[0] / len(df) * 100 if not df[selected_cat_col].value_counts().empty else 0,
                            df[selected_cat_col].isnull().sum(),
                            df[selected_cat_col].isnull().sum() / len(df) * 100
                        ]
                    })
                    st.dataframe(cat_stats, use_container_width=True)
                
                # Visualizations
                st.subheader(f"Visualization for {selected_cat_col}")
                
                # Bar chart for categorical column
                fig, ax = plt.subplots(figsize=(12, min(8, len(value_counts) * 0.4)))
                bars = sns.barplot(x='Count', y=selected_cat_col, data=value_counts, ax=ax)
                
                # Add percentage annotations to bars
                for i, p in enumerate(bars.patches):
                    percentage = value_counts.iloc[i]['Percentage']
                    plt.text(p.get_width() + 0.5, p.get_y() + p.get_height()/2, 
                            f'{percentage:.1f}%', 
                            ha='left', va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Pie chart for top categories
                fig, ax = plt.subplots(figsize=(8, 8))
                pie_data = value_counts.head(7)  # Top 7 categories
                
                # Add "Others" category for the rest
                if len(value_counts) > 7:
                    others_count = value_counts['Count'][7:].sum()
                    others_row = pd.DataFrame({selected_cat_col: ['Others'], 'Count': [others_count], 
                                            'Percentage': [value_counts['Percentage'][7:].sum()]})
                    pie_data = pd.concat([pie_data, others_row])
                
                plt.pie(pie_data['Count'], labels=pie_data[selected_cat_col], autopct='%1.1f%%',
                        startangle=90, shadow=True)
                plt.axis('equal')
                plt.title(f'Distribution of {selected_cat_col} (Top Categories)')
                st.pyplot(fig)
        else:
            st.info("No categorical columns found in the dataset.")
    
    with tab3:
        numeric_cols = get_numeric_columns(df)
        if len(numeric_cols) > 1:
            st.subheader("Correlation Analysis")
            
            # Use utility function for correlation heatmap
            corr_fig = plot_correlation_heatmap(df, numeric_cols)
            st.pyplot(corr_fig)
            
            # Correlation statistics
            st.subheader("Highest Correlations")
            
            # Find highest absolute correlations
            corr_matrix = df[numeric_cols].corr().round(2)
            correlations = corr_matrix.unstack().sort_values(ascending=False)
            # Remove self-correlations
            high_correlations = correlations[correlations < 1].head(10)
            
            if not high_correlations.empty:
                corr_df = pd.DataFrame(high_correlations).reset_index()
                corr_df.columns = ['Variable 1', 'Variable 2', 'Correlation']
                st.dataframe(corr_df, use_container_width=True)
                
                # Scatter plot for highest correlation pair
                if len(high_correlations) > 0:
                    var1 = corr_df.iloc[0]['Variable 1']
                    var2 = corr_df.iloc[0]['Variable 2']
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(x=df[var1], y=df[var2], ax=ax)
                    plt.title(f'Scatter Plot: {var1} vs {var2}')
                    st.pyplot(fig)
            else:
                st.info("No significant correlations found between numeric variables.")
        else:
            st.info("Need at least two numeric columns for correlation analysis.")
    
    # Link to description
    with st.expander("What do the summary statistics fields mean?", expanded=False):
        st.markdown("""
        ### Numeric Statistics
        - **count**: Number of non-null entries in the column
        - **mean**: Average value (sum of values divided by count)
        - **std**: Standard deviation (measure of data spread)
        - **min**: Minimum value
        - **1%**: 1st percentile (value below which 1% of observations are found)
        - **5%**: 5th percentile (value below which 5% of observations are found)
        - **25%**: First quartile (value below which 25% of observations are found)
        - **50%**: Median/second quartile (middle value)
        - **75%**: Third quartile (value below which 75% of observations are found)
        - **95%**: 95th percentile (value below which 95% of observations are found)
        - **99%**: 99th percentile (value below which 99% of observations are found)
        - **max**: Maximum value
        - **skew**: Measure of asymmetry of the distribution
        - **kurtosis**: Measure of the "tailedness" of the distribution
        - **missing**: Count of missing values
        - **missing %**: Percentage of missing values
        
        ### Categorical Statistics
        - **unique**: Number of unique values
        - **top**: Most frequent value
        - **freq**: Frequency of the most frequent value
        """)
else:
    show_file_required_warning()

# Footer
create_footer() 
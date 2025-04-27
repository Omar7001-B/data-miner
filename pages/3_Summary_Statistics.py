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

# Initialize runpage in session state if not present
if 'runpage' not in st.session_state:
    st.session_state.runpage = None

if st.session_state.runpage == "3_Summary_Statistics" or st.session_state.runpage is None:
    st.session_state.runpage = None
    st.set_page_config(page_title="Summary Statistics", page_icon="📈", layout="wide")
    load_css()

    st.title("Summary Statistics 📈")
    st.markdown("Get comprehensive statistics and insights about your dataset's values.")

    df = st.session_state.get('df', None)
    if df is not None:
        # Display dataset metrics
        display_dataset_info()
        
        # Initialize session state variables
        if 'rows_per_page' not in st.session_state:
            st.session_state.rows_per_page = 10
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1
        if 'select_all_clicked' not in st.session_state:
            st.session_state.select_all_clicked = False
        if 'clear_all_clicked' not in st.session_state:
            st.session_state.clear_all_clicked = False
            
        def handle_select_all():
            st.session_state.select_all_clicked = True
            st.session_state.clear_all_clicked = False
            
        def handle_clear_all():
            st.session_state.clear_all_clicked = True
            st.session_state.select_all_clicked = False
        
        # Create tabs for different types of statistics
        tab0, tab1, tab2, tab3 = st.tabs(["Column Selection", "Numeric Stats", "Categorical Stats", "Correlation Analysis"])
        
        with tab0:
            # Main container for column selection
            with st.container():
                st.subheader("🔍 Column Selection")
                st.markdown("Select the columns you want to analyze from your dataset.")
                
                # Column selection area
                col1, col2 = st.columns([3, 1])
                
                # Determine default value for multiselect based on button clicks
                if st.session_state.select_all_clicked:
                    default_selection = df.columns.tolist()
                    st.session_state.select_all_clicked = False
                elif st.session_state.clear_all_clicked:
                    default_selection = []
                    st.session_state.clear_all_clicked = False
                else:
                    default_selection = df.columns.tolist() if 'column_selector' not in st.session_state else st.session_state.column_selector
                
                with col1:
                    selected_columns = st.multiselect(
                        "Select columns to analyze:",
                        options=df.columns.tolist(),
                        default=default_selection,
                        key="column_selector",
                        help="You can select multiple columns by clicking or searching"
                    )
                
                with col2:
                    st.markdown("##### Quick Actions")
                    col_count = len(selected_columns)
                    st.caption(f"Selected: {col_count} of {len(df.columns)} columns")
                    
                    # Quick selection buttons
                    st.button("Select All Columns", 
                             on_click=handle_select_all,
                             type="primary",
                             use_container_width=True)
                    
                    st.button("Clear Selection", 
                             on_click=handle_clear_all,
                             use_container_width=True)

            # Data preview section
            if selected_columns:
                st.divider()
                
                with st.container():
                    st.subheader("📊 Data Preview")
                    
                    # Calculate pagination info
                    total_rows = len(df)
                    total_pages = (total_rows - 1) // st.session_state.rows_per_page + 1
                    start_idx = (st.session_state.current_page - 1) * st.session_state.rows_per_page
                    end_idx = min(start_idx + st.session_state.rows_per_page, total_rows)
                    
                    # Pagination controls in a single row
                    col1, col2, col3 = st.columns([1.2, 1.5, 1.2])
                    
                    with col1:
                        st.selectbox(
                            "Rows per page",
                            options=[5, 10, 20, 50, 100],
                            index=1,
                            key="rows_per_page_selector",
                            help="Select number of rows to display per page"
                        )
                        st.session_state.rows_per_page = st.session_state.rows_per_page_selector
                    
                    with col2:
                        page_number = st.number_input(
                            "Page",
                            min_value=1,
                            max_value=total_pages,
                            value=st.session_state.current_page,
                            help=f"Enter a page number between 1 and {total_pages}"
                        )
                        if page_number != st.session_state.current_page:
                            st.session_state.current_page = page_number
                            st.rerun()
                    
                    with col3:
                        st.markdown(f"Page {st.session_state.current_page} of {total_pages}")
                        st.caption(f"Showing {start_idx + 1} to {end_idx} of {total_rows} rows")

                    # Display data preview
                    st.dataframe(
                        df[selected_columns].iloc[start_idx:end_idx],
                        use_container_width=True,
                        hide_index=False
                    )
                    
                    # Navigation buttons under the table
                    nav_col1, nav_col2, nav_col3 = st.columns([1, 4, 1])
                    with nav_col1:
                        if st.button("◀ Previous", disabled=(st.session_state.current_page == 1), use_container_width=True):
                            st.session_state.current_page -= 1
                            st.rerun()
                    with nav_col3:
                        if st.button("Next ▶", disabled=(st.session_state.current_page == total_pages), use_container_width=True):
                            st.session_state.current_page += 1
                            st.rerun()

                # Column information section
                st.divider()
                with st.container():
                    st.subheader("📋 Column Information")
                    
                    # Create column information table
                    col_info = []
                    for col in selected_columns:
                        non_null_count = df[col].count()
                        null_count = df[col].isnull().sum()
                        unique_count = df[col].nunique()
                        
                        col_info.append({
                            "Column Name": col,
                            "Data Type": str(df[col].dtype),
                            "Non-Null Values": f"{non_null_count:,} ({(non_null_count/len(df))*100:.1f}%)",
                            "Null Values": f"{null_count:,} ({(null_count/len(df))*100:.1f}%)",
                            "Unique Values": f"{unique_count:,}",
                            "Memory Usage": f"{df[col].memory_usage(deep=True)/1024:,.1f} KB"
                        })
                    
                    col_info_df = pd.DataFrame(col_info)
                    st.dataframe(
                        col_info_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Column Name": st.column_config.TextColumn("Column Name", help="Name of the column in the dataset"),
                            "Data Type": st.column_config.TextColumn("Data Type", help="The data type of the column"),
                            "Non-Null Values": st.column_config.TextColumn("Non-Null Values", help="Count and percentage of non-null values"),
                            "Null Values": st.column_config.TextColumn("Null Values", help="Count and percentage of null values"),
                            "Unique Values": st.column_config.TextColumn("Unique Values", help="Count of unique values in the column"),
                            "Memory Usage": st.column_config.TextColumn("Memory Usage", help="Memory used by the column")
                        }
                    )
            else:
                st.info("👆 Please select at least one column to view the data preview and column information.")

        # Use selected columns for analysis in other tabs
        if selected_columns:
            df_selected = df[selected_columns]
            
            with tab1:
                numeric_cols = get_numeric_columns(df_selected)
                if len(numeric_cols) > 0:
                    st.subheader("Numeric Column Statistics")
                    
                    # Column selector for numeric columns
                    cols_to_analyze = st.multiselect(
                        "Select numeric columns to analyze:",
                        options=numeric_cols,
                        default=list(numeric_cols[:min(5, len(numeric_cols))])
                    )
                    
                    if cols_to_analyze:
                        # Use the utility function for summary stats
                        stats_df = generate_summary_stats(df_selected, cols_to_analyze)
                        st.dataframe(stats_df.round(2), use_container_width=True)
                        
                        # Visualizations
                        st.subheader("Distribution Visualization")
                        
                        # Box plot
                        fig, ax = plt.subplots(figsize=(12, 5))
                        sns.boxplot(data=df_selected[cols_to_analyze], ax=ax)
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
                                    sns.histplot(df_selected[col].dropna(), kde=True, ax=axes[i])
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
                    st.info("No numeric columns found in the selected columns.")
            
            with tab2:
                categorical_cols = df_selected.select_dtypes(include=["object", "category"]).columns
                if len(categorical_cols) > 0:
                    st.subheader("Categorical Column Statistics")
                    
                    # Column selector
                    selected_cat_col = st.selectbox("Select a categorical column:", categorical_cols)
                    
                    if selected_cat_col:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Enhanced categorical stats with percentages
                            value_counts = df_selected[selected_cat_col].value_counts().reset_index()
                            value_counts.columns = [selected_cat_col, 'Count']
                            value_counts['Percentage'] = (value_counts['Count'] / len(df_selected)) * 100
                            
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
                                    len(df_selected[selected_cat_col]),
                                    df_selected[selected_cat_col].nunique(),
                                    df_selected[selected_cat_col].value_counts().index[0] if not df_selected[selected_cat_col].value_counts().empty else None,
                                    df_selected[selected_cat_col].value_counts().iloc[0] if not df_selected[selected_cat_col].value_counts().empty else 0,
                                    df_selected[selected_cat_col].value_counts().iloc[0] / len(df_selected) * 100 if not df_selected[selected_cat_col].value_counts().empty else 0,
                                    df_selected[selected_cat_col].isnull().sum(),
                                    df_selected[selected_cat_col].isnull().sum() / len(df_selected) * 100
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
                    st.info("No categorical columns found in the selected columns.")
            
            with tab3:
                numeric_cols = get_numeric_columns(df_selected)
                if len(numeric_cols) > 1:
                    st.subheader("Correlation Analysis")
                    
                    # Use utility function for correlation heatmap
                    corr_fig = plot_correlation_heatmap(df_selected, numeric_cols)
                    st.pyplot(corr_fig)
                    
                    # Correlation statistics
                    st.subheader("Highest Correlations")
                    
                    # Find highest absolute correlations
                    corr_matrix = df_selected[numeric_cols].corr().round(2)
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
                            sns.scatterplot(x=df_selected[var1], y=df_selected[var2], ax=ax)
                            plt.title(f'Scatter Plot: {var1} vs {var2}')
                            st.pyplot(fig)
                    else:
                        st.info("No significant correlations found between numeric columns.")
                else:
                    st.info("Please select at least two numeric columns for correlation analysis.")
    else:
        show_file_required_warning()

    # Add footer
    create_footer() 
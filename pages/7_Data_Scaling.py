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
    get_numeric_columns
)

st.set_page_config(page_title="Data Scaling", page_icon="📊", layout="wide")
load_css()

st.title("Data Scaling and Normalization 📊")
st.markdown("Apply various scaling methods to normalize your numerical data for machine learning models.")

df = st.session_state.get('df', None)
if df is not None:
    # Display dataset metrics
    display_dataset_info()
    
    # Get numeric columns
    numeric_columns = get_numeric_columns(df)
    
    if not numeric_columns:
        create_card(
            "No Numeric Columns",
            "Your dataset doesn't have any numeric columns that need scaling.",
            icon="ℹ️"
        )
    else:
        # Create tabs for different aspects
        tab1, tab2, tab3 = st.tabs(["Overview", "Scaling Methods", "Apply Scaling"])

        with tab1:
            st.subheader("Numeric Columns Overview")
            
            # Display metrics about numeric columns
            metrics_cols = st.columns(3)
            with metrics_cols[0]:
                st.metric("Total Columns", f"{df.shape[1]}")
            with metrics_cols[1]:
                st.metric("Numeric Columns", f"{len(numeric_columns)}")
            with metrics_cols[2]:
                num_percent = (len(numeric_columns) / df.shape[1]) * 100
                st.metric("Percentage Numeric", f"{num_percent:.1f}%")
            
            # Display numeric columns and their statistics
            st.subheader("Numeric Columns Statistics")
            
            # Create a DataFrame with statistics about numeric columns
            num_stats = pd.DataFrame(index=numeric_columns)
            num_stats['Min'] = df[numeric_columns].min()
            num_stats['Max'] = df[numeric_columns].max()
            num_stats['Range'] = num_stats['Max'] - num_stats['Min']
            num_stats['Mean'] = df[numeric_columns].mean()
            num_stats['Median'] = df[numeric_columns].median()
            num_stats['Std Dev'] = df[numeric_columns].std()
            num_stats['CV (%)'] = (num_stats['Std Dev'] / num_stats['Mean'].abs()) * 100
            
            # Add missing values
            num_stats['Missing'] = df[numeric_columns].isnull().sum()
            num_stats['Missing %'] = (df[numeric_columns].isnull().sum() / len(df)) * 100
            
            # Reset index to make column name accessible
            num_stats = num_stats.reset_index().rename(columns={'index': 'Column'})
            
            # Style the DataFrame
            def highlight_large_range(val):
                if isinstance(val, (int, float)) and pd.notnull(val):
                    if val > 1000:
                        return 'background-color: #FFF7B2'
                return ''
            
            def highlight_high_cv(val):
                if isinstance(val, (int, float)) and pd.notnull(val):
                    if val > 100:
                        return 'background-color: #FFD1D1'
                return ''
            
            styled_df = num_stats.style.applymap(
                highlight_large_range, subset=['Range']
            ).applymap(
                highlight_high_cv, subset=['CV (%)']
            ).background_gradient(cmap='YlOrRd', subset=['Missing %'])
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Visualize distributions
            st.subheader("Distribution of Numeric Features")
            
            # Allow user to select columns to visualize
            selected_cols = st.multiselect(
                "Select columns to visualize",
                options=numeric_columns,
                default=numeric_columns[:min(3, len(numeric_columns))]
            )
            
            if selected_cols:
                # Create a figure with subplots
                fig, axes = plt.subplots(len(selected_cols), 2, figsize=(14, 4 * len(selected_cols)))
                
                # Handle case of single column
                if len(selected_cols) == 1:
                    axes = axes.reshape(1, 2)
                
                for i, col in enumerate(selected_cols):
                    # Histogram on the left
                    sns.histplot(df[col].dropna(), kde=True, ax=axes[i, 0], color='cornflowerblue')
                    axes[i, 0].set_title(f'Distribution of {col}')
                    axes[i, 0].grid(True, alpha=0.3)
                    
                    # Box plot on the right
                    sns.boxplot(x=df[col].dropna(), ax=axes[i, 1], color='cornflowerblue')
                    axes[i, 1].set_title(f'Box Plot of {col}')
                    axes[i, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Scaling recommendations
            st.subheader("Scaling Recommendations")
            
            # Check ranges and distributions
            wide_range = [col for col in numeric_columns if (df[col].max() - df[col].min()) > 100]
            skewed = [col for col in numeric_columns if abs(df[col].skew()) > 1]
            outliers = []
            
            for col in numeric_columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                upper_bound = q3 + 1.5 * iqr
                lower_bound = q1 - 1.5 * iqr
                
                if (df[col] > upper_bound).any() or (df[col] < lower_bound).any():
                    outliers.append(col)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Min-Max Scaling Candidates")
                if wide_range:
                    st.markdown("These columns have wide ranges and are good candidates for min-max scaling:")
                    for col in wide_range:
                        min_val = df[col].min()
                        max_val = df[col].max()
                        st.markdown(f"- **{col}**: Range from {min_val:.2f} to {max_val:.2f}")
                else:
                    st.info("No columns with particularly wide ranges found.")
            
            with col2:
                st.markdown("#### Z-Score Normalization Candidates")
                if outliers:
                    st.markdown("These columns have outliers and might benefit from z-score normalization:")
                    for col in outliers:
                        std = df[col].std()
                        mean = df[col].mean()
                        st.markdown(f"- **{col}**: Mean = {mean:.2f}, Std Dev = {std:.2f}")
                else:
                    st.info("No columns with significant outliers found.")

        with tab2:
            st.subheader("Scaling Methods Explained")
            
            # Create columns for different scaling methods
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Min-Max Scaling")
                st.markdown("""
                **What it does:**
                - Scales data to a fixed range - usually 0 to 1
                - Formula: (X - X_min) / (X_max - X_min)
                
                **When to use:**
                - When you need values in a specific range
                - For algorithms sensitive to feature magnitudes
                - When the distribution is not Gaussian or unknown
                
                **Advantages:**
                - Preserves relationships in the data
                - All features have the same scale
                
                **Disadvantages:**
                - Sensitive to outliers
                - Doesn't reduce the effect of outliers
                """)
                
                # Example of min-max scaling
                if numeric_columns:
                    example_col = numeric_columns[0]
                    
                    st.markdown(f"**Example with column '{example_col}':**")
                    
                    sample_df = pd.DataFrame({
                        'Original': df[example_col].dropna().head(5)
                    })
                    
                    # Calculate min-max scaled values
                    min_val = df[example_col].min()
                    max_val = df[example_col].max()
                    sample_df['Min-Max Scaled'] = (sample_df['Original'] - min_val) / (max_val - min_val)
                    
                    st.dataframe(sample_df.style.format({'Original': '{:.2f}', 'Min-Max Scaled': '{:.2f}'}))
            
            with col2:
                st.markdown("### Z-Score Normalization")
                st.markdown("""
                **What it does:**
                - Scales data to have mean = 0, standard deviation = 1
                - Formula: (X - μ) / σ
                
                **When to use:**
                - When the data follows a normal distribution
                - When dealing with outliers
                - For algorithms using Euclidean distance
                
                **Advantages:**
                - Handles outliers better than min-max scaling
                - Useful for comparing features with different units
                
                **Disadvantages:**
                - Doesn't produce values in a bounded range
                - Less interpretable than min-max scaling
                """)
                
                # Example of z-score normalization
                if numeric_columns:
                    example_col = numeric_columns[0]
                    
                    st.markdown(f"**Example with column '{example_col}':**")
                    
                    sample_df = pd.DataFrame({
                        'Original': df[example_col].dropna().head(5)
                    })
                    
                    # Calculate z-score scaled values
                    mean_val = df[example_col].mean()
                    std_val = df[example_col].std()
                    sample_df['Z-Score Normalized'] = (sample_df['Original'] - mean_val) / std_val
                    
                    st.dataframe(sample_df.style.format({'Original': '{:.2f}', 'Z-Score Normalized': '{:.2f}'}))
            
            with col3:
                st.markdown("### Decimal Scaling")
                st.markdown("""
                **What it does:**
                - Scales by dividing by a power of 10
                - Formula: X / 10^j (where j is the smallest integer such that max(|X/10^j|) < 1)
                
                **When to use:**
                - When you want to preserve the general distribution
                - For simpler scaling needs
                - When interpretability is important
                
                **Advantages:**
                - Very simple to understand and implement
                - Preserves the distribution shape
                
                **Disadvantages:**
                - Limited control over the final range
                - May not fully normalize the data
                """)
                
                # Example of decimal scaling
                if numeric_columns:
                    example_col = numeric_columns[0]
                    
                    st.markdown(f"**Example with column '{example_col}':**")
                    
                    sample_df = pd.DataFrame({
                        'Original': df[example_col].dropna().head(5)
                    })
                    
                    # Calculate decimal scaled values
                    max_abs = df[example_col].abs().max()
                    j = int(np.log10(max_abs)) + 1
                    sample_df['Decimal Scaled'] = sample_df['Original'] / (10 ** j)
                    
                    st.dataframe(sample_df.style.format({'Original': '{:.2f}', 'Decimal Scaled': '{:.6f}'}))

        with tab3:
            st.subheader("Apply Data Scaling")
            
            # Select columns to scale
            if len(numeric_columns) > 0:
                st.markdown("### Step 1: Select Columns to Scale")
                
                selected_numeric_cols = st.multiselect(
                    "Select numeric columns to scale",
                    options=numeric_columns,
                    default=numeric_columns[:min(3, len(numeric_columns))],
                    help="Only numeric columns can be scaled"
                )
                
                if not selected_numeric_cols:
                    st.warning("⚠️ Please select at least one column to scale.")
                
                # Select scaling method
                st.markdown("### Step 2: Choose Scaling Method")
                
                scaling_method = st.radio(
                    "Select scaling method",
                    ["Min-Max Scaling", "Z-Score Normalization", "Decimal Scaling"],
                    help="Each method has different properties and use cases"
                )
                
                # Additional options
                st.markdown("### Step 3: Configure Options")
                
                options_col1, options_col2 = st.columns(2)
                
                with options_col1:
                    if scaling_method == "Min-Max Scaling":
                        min_value = st.number_input("Custom Minimum Value", value=0.0)
                        max_value = st.number_input("Custom Maximum Value", value=1.0)
                    elif scaling_method == "Z-Score Normalization":
                        handle_outliers = st.checkbox(
                            "Clip extreme values (> 3 standard deviations)",
                            value=False,
                            help="Limits extreme values after normalization"
                        )
                    else:  # Decimal Scaling
                        custom_power = st.checkbox(
                            "Use custom power of 10",
                            value=False,
                            help="Specify a custom power instead of calculating automatically"
                        )
                        if custom_power:
                            power = st.number_input("Power of 10", min_value=1, max_value=10, value=1)
                
                with options_col2:
                    handle_missing = st.radio(
                        "How to handle missing values?",
                        ["Skip (leave as NaN)", "Fill with mean before scaling"],
                        help="Missing values can cause issues with scaling"
                    )
                    
                    append_suffix = st.checkbox(
                        "Append suffix to scaled columns",
                        value=True,
                        help="Adds a suffix to column names to indicate scaling"
                    )
                
                # Apply scaling button
                if st.button("Apply Scaling", disabled=not selected_numeric_cols):
                    if not selected_numeric_cols:
                        st.warning("Please select at least one column to scale.")
                    else:
                        with st.spinner(f"Applying {scaling_method} to selected columns..."):
                            # Make a copy of the DataFrame
                            df_scaled = df.copy()
                            
                            # Process each selected column
                            for col in selected_numeric_cols:
                                # Handle missing values if specified
                                if handle_missing == "Fill with mean before scaling":
                                    df_scaled[col] = df_scaled[col].fillna(df_scaled[col].mean())
                                
                                # Apply the appropriate scaling method
                                if scaling_method == "Min-Max Scaling":
                                    # Get column min and max
                                    col_min = df_scaled[col].min()
                                    col_max = df_scaled[col].max()
                                    
                                    # Avoid division by zero
                                    if col_max > col_min:
                                        new_col_name = f"{col}_minmax" if append_suffix else col
                                        df_scaled[new_col_name] = min_value + (df_scaled[col] - col_min) / (col_max - col_min) * (max_value - min_value)
                                        
                                        # If not appending suffix, remove the original column
                                        if not append_suffix and new_col_name != col:
                                            df_scaled = df_scaled.drop(columns=[col])
                                    else:
                                        st.warning(f"Column {col} has min = max. Skipping scaling.")
                                        
                                elif scaling_method == "Z-Score Normalization":
                                    # Get column mean and std
                                    col_mean = df_scaled[col].mean()
                                    col_std = df_scaled[col].std()
                                    
                                    # Avoid division by zero
                                    if col_std > 0:
                                        new_col_name = f"{col}_zscore" if append_suffix else col
                                        df_scaled[new_col_name] = (df_scaled[col] - col_mean) / col_std
                                        
                                        # Clip extreme values if specified
                                        if handle_outliers:
                                            df_scaled[new_col_name] = df_scaled[new_col_name].clip(-3, 3)
                                            
                                        # If not appending suffix, remove the original column
                                        if not append_suffix and new_col_name != col:
                                            df_scaled = df_scaled.drop(columns=[col])
                                    else:
                                        st.warning(f"Column {col} has standard deviation = 0. Skipping scaling.")
                                        
                                else:  # Decimal Scaling
                                    # Determine the power of 10 to divide by
                                    if custom_power:
                                        j = power
                                    else:
                                        max_abs = df_scaled[col].abs().max()
                                        j = int(np.log10(max_abs)) + 1 if max_abs > 0 else 1
                                    
                                    new_col_name = f"{col}_decimal" if append_suffix else col
                                    df_scaled[new_col_name] = df_scaled[col] / (10 ** j)
                                    
                                    # If not appending suffix, remove the original column
                                    if not append_suffix and new_col_name != col:
                                        df_scaled = df_scaled.drop(columns=[col])
                            
                            # Store the result
                            st.session_state['df_scaled'] = df_scaled
                            
                            st.success(f"✅ Successfully applied {scaling_method} to {len(selected_numeric_cols)} columns!")
                            
                            # Show download button
                            st.download_button(
                                "📥 Download Scaled Data as CSV", 
                                df_scaled.to_csv(index=False), 
                                file_name="scaled_data.csv",
                                mime="text/csv"
                            )
                            
                            # Offer option to replace original dataframe
                            if st.button("📥 Replace Original Data with Scaled Version"):
                                st.session_state['df'] = df_scaled.copy()
                                st.success("Original data replaced with scaled version!")
                                st.rerun()
                
                # Show result if available
                if 'df_scaled' in st.session_state:
                    with st.expander("Preview Scaled Data", expanded=True):
                        st.dataframe(st.session_state['df_scaled'].head(10), use_container_width=True)
                        
                        # Show before/after distributions for a selected column
                        if 'df_scaled' in st.session_state and selected_numeric_cols:
                            st.markdown("### Before/After Scaling Comparison")
                            
                            # Let user select a column to view
                            compare_col = st.selectbox(
                                "Select column to compare before/after scaling",
                                options=selected_numeric_cols
                            )
                            
                            if compare_col:
                                # Determine the name of the scaled column
                                if scaling_method == "Min-Max Scaling" and append_suffix:
                                    scaled_col = f"{compare_col}_minmax"
                                elif scaling_method == "Z-Score Normalization" and append_suffix:
                                    scaled_col = f"{compare_col}_zscore"
                                elif scaling_method == "Decimal Scaling" and append_suffix:
                                    scaled_col = f"{compare_col}_decimal"
                                else:
                                    scaled_col = compare_col
                                
                                # Check if the scaled column exists
                                if scaled_col in st.session_state['df_scaled'].columns:
                                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                                    
                                    # Original distribution
                                    sns.histplot(df[compare_col].dropna(), kde=True, ax=axes[0], color='cornflowerblue')
                                    axes[0].set_title(f'Original Distribution of {compare_col}')
                                    axes[0].grid(True, alpha=0.3)
                                    
                                    # Scaled distribution
                                    sns.histplot(st.session_state['df_scaled'][scaled_col].dropna(), kde=True, ax=axes[1], color='green')
                                    axes[1].set_title(f'Scaled Distribution of {scaled_col}')
                                    axes[1].grid(True, alpha=0.3)
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    
                                    # Display statistics for comparison
                                    stats_col1, stats_col2 = st.columns(2)
                                    with stats_col1:
                                        st.markdown(f"**Original {compare_col} Statistics**")
                                        st.dataframe(df[compare_col].describe())
                                    with stats_col2:
                                        st.markdown(f"**Scaled {scaled_col} Statistics**")
                                        st.dataframe(st.session_state['df_scaled'][scaled_col].describe())
            else:
                create_card(
                    "No Numeric Columns",
                    "Your dataset doesn't have any numeric columns that need scaling.",
                    icon="ℹ️"
                )
else:
    show_file_required_warning()

# Footer
create_footer() 
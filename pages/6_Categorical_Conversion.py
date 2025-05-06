import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from utils import (
    load_css, 
    create_footer, 
    show_file_required_warning, 
    display_dataset_info, 
    create_card,
    get_categorical_columns
)

st.set_page_config(page_title="Categorical Conversion", page_icon="🔄", layout="wide")
load_css()

st.title("Categorical Values Conversion 🔄")
st.markdown("Convert categorical values to numerical format for machine learning models using encoding techniques.")

df = st.session_state.get('df', None)
if df is not None:
    # Display dataset metrics
    display_dataset_info()
    
    # Get categorical columns
    categorical_columns = get_categorical_columns(df)
    
    if not categorical_columns:
        create_card(
            "No Categorical Columns",
            "Your dataset doesn't have any categorical columns that need conversion.",
            icon="ℹ️"
        )
    else:
        # Create tabs for different aspects
        tab1, tab2, tab3 = st.tabs(["Overview", "Encoding Methods", "Apply Conversion"])

        with tab1:
            st.subheader("Categorical Columns Overview")
            
            # Display metrics about categorical columns
            metrics_cols = st.columns(3)
            with metrics_cols[0]:
                st.metric("Total Columns", f"{df.shape[1]}")
            with metrics_cols[1]:
                st.metric("Categorical Columns", f"{len(categorical_columns)}")
            with metrics_cols[2]:
                cat_percent = (len(categorical_columns) / df.shape[1]) * 100
                st.metric("Percentage Categorical", f"{cat_percent:.1f}%")
            
            # Display categorical columns and their details
            st.subheader("Categorical Columns Details")
            
            # Create a DataFrame with information about categorical columns
            cat_info = []
            for col in categorical_columns:
                unique_values = df[col].nunique()
                unique_examples = ", ".join(df[col].dropna().sample(min(3, unique_values)).astype(str).tolist())
                missing_values = df[col].isnull().sum()
                missing_percent = (missing_values / len(df)) * 100
                
                cat_info.append({
                    "Column": col,
                    "Unique Values": unique_values,
                    "Top Values (examples)": unique_examples,
                    "Missing Values": missing_values,
                    "Missing %": missing_percent
                })
            
            cat_info_df = pd.DataFrame(cat_info)
            
            # Style the DataFrame
            def highlight_many_values(val):
                if isinstance(val, int) and val > 10:
                    return 'background-color: #FFF7B2'
                return ''
            
            styled_df = cat_info_df.style.applymap(
                highlight_many_values, subset=['Unique Values']
            ).background_gradient(cmap='YlOrRd', subset=['Missing %'])
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Display distribution of unique values
            st.subheader("Distribution of Unique Values")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot bar chart of unique values count
            unique_counts = [df[col].nunique() for col in categorical_columns]
            sns.barplot(x=categorical_columns, y=unique_counts, ax=ax, palette='viridis')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_ylabel('Number of Unique Values')
            ax.set_title('Unique Values per Categorical Column')
            
            # Add text labels
            for i, count in enumerate(unique_counts):
                ax.text(i, count + 0.5, str(count), ha='center')
                
            plt.tight_layout()
            st.pyplot(fig)
            
            # Recommendations based on unique values
            st.subheader("Encoding Recommendations")
            
            high_cardinality = [col for col in categorical_columns if df[col].nunique() > 10]
            low_cardinality = [col for col in categorical_columns if df[col].nunique() <= 10]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### One-Hot Encoding Candidates")
                if low_cardinality:
                    st.markdown("These columns have few unique values and are good candidates for one-hot encoding:")
                    for col in low_cardinality:
                        unique_count = df[col].nunique()
                        st.markdown(f"- **{col}**: {unique_count} unique values")
                else:
                    st.info("No columns with low cardinality found.")
            
            with col2:
                st.markdown("#### Label Encoding Candidates")
                if high_cardinality:
                    st.markdown("These columns have many unique values and may be better suited for label encoding:")
                    for col in high_cardinality:
                        unique_count = df[col].nunique()
                        st.markdown(f"- **{col}**: {unique_count} unique values")
                else:
                    st.info("No columns with high cardinality found.")

        with tab2:
            st.subheader("Encoding Methods Explained")
            
            # Create columns for different encoding methods
            method_col1, method_col2 = st.columns(2)
            
            with method_col1:
                st.markdown("### One-Hot Encoding")
                st.markdown("""
                **What it does:**
                - Creates a new binary column for each unique value
                - Each row has a 1 in exactly one of these new columns
                - Original column is removed
                
                **When to use:**
                - When categories have no ordinal relationship
                - When there are relatively few unique values
                - For algorithms sensitive to numeric magnitude (like neural networks)
                
                **Advantages:**
                - Prevents algorithm from assuming ordinal relationships
                - Works well with most ML algorithms
                
                **Disadvantages:**
                - Creates many new columns if there are many unique values
                - Can lead to sparse matrices
                """)
                
                # Example of one-hot encoding
                if categorical_columns:
                    example_col = categorical_columns[0]
                    unique_vals = min(df[example_col].nunique(), 5)
                    
                    st.markdown(f"**Example with column '{example_col}':**")
                    
                    sample_df = df[[example_col]].head(5).reset_index(drop=True)
                    dummies = pd.get_dummies(sample_df[example_col], prefix=example_col)
                    encoded_df = pd.concat([sample_df, dummies], axis=1)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original:**")
                        st.dataframe(sample_df)
                    with col2:
                        st.markdown("**One-Hot Encoded:**")
                        st.dataframe(encoded_df)
            
            with method_col2:
                st.markdown("### Label Encoding")
                st.markdown("""
                **What it does:**
                - Converts each unique value to an integer
                - Preserves a single column structure
                
                **When to use:**
                - When there are many unique values
                - When there is an ordinal relationship (small, medium, large)
                - For tree-based models (Random Forest, Decision Trees)
                
                **Advantages:**
                - Maintains the original number of columns
                - More memory-efficient than one-hot encoding
                
                **Disadvantages:**
                - Introduces ordinal relationships that may not exist
                - May not work well with distance-based algorithms
                """)
                
                # Example of label encoding
                if categorical_columns:
                    example_col = categorical_columns[0]
                    
                    st.markdown(f"**Example with column '{example_col}':**")
                    
                    sample_df = df[[example_col]].head(5).reset_index(drop=True)
                    le = LabelEncoder()
                    sample_df['Encoded'] = le.fit_transform(sample_df[example_col])
                    
                    # Create mapping table
                    mapping = pd.DataFrame({
                        'Original': le.classes_,
                        'Encoded': range(len(le.classes_))
                    }).head(min(len(le.classes_), 10))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original vs Encoded:**")
                        st.dataframe(sample_df)
                    with col2:
                        st.markdown("**Mapping:**")
                        st.dataframe(mapping)

        with tab3:
            st.subheader("Apply Categorical Conversion")
            
            # Select columns to encode
            if len(categorical_columns) > 0:
                st.markdown("### Step 1: Select Columns to Convert")
                
                # Create two columns for selection
                select_col1, select_col2 = st.columns(2)
                
                with select_col1:
                    # Select columns for one-hot encoding
                    onehot_columns = st.multiselect(
                        "Select columns for One-Hot Encoding",
                        options=categorical_columns,
                        default=[col for col in low_cardinality if len(low_cardinality) <= 3],
                        help="Best for columns with few unique values"
                    )
                
                with select_col2:
                    # Select columns for label encoding
                    label_columns = st.multiselect(
                        "Select columns for Label Encoding",
                        options=[col for col in categorical_columns if col not in onehot_columns],
                        default=[col for col in high_cardinality if len(high_cardinality) <= 3],
                        help="Best for columns with many unique values or ordinal data"
                    )
                
                # Options for handling missing values
                st.markdown("### Step 2: Configure Options")
                
                options_col1, options_col2 = st.columns(2)
                
                with options_col1:
                    handle_missing = st.radio(
                        "How to handle missing values?",
                        ["Fill with 'unknown'", "Drop rows with missing values"],
                        help="Missing values can cause issues with encoding"
                    )
                
                with options_col2:
                    handle_new_cats = st.radio(
                        "How to handle new categories at prediction time?",
                        ["Use robust encoding (recommended)", "Standard encoding"],
                        help="Robust encoding handles unseen categories better for production"
                    )
                
                # Apply conversion button
                if st.button("Apply Conversion", disabled=not (onehot_columns or label_columns)):
                    if not (onehot_columns or label_columns):
                        st.warning("Please select at least one column to convert.")
                    else:
                        with st.spinner("Converting categorical values..."):
                            # Make a copy of the DataFrame
                            df_converted = df.copy()
                            
                            # Handle missing values
                            if handle_missing == "Fill with 'unknown'":
                                for col in onehot_columns + label_columns:
                                    df_converted[col] = df_converted[col].fillna('unknown')
                            else:  # Drop rows
                                cols_with_na = [col for col in onehot_columns + label_columns if df_converted[col].isnull().any()]
                                if cols_with_na:
                                    before_count = len(df_converted)
                                    df_converted = df_converted.dropna(subset=cols_with_na)
                                    dropped_count = before_count - len(df_converted)
                                    st.info(f"Dropped {dropped_count} rows with missing values.")
                            
                            # Process columns for one-hot encoding
                            if onehot_columns:
                                # Get dummy variables
                                if handle_new_cats == "Use robust encoding (recommended)":
                                    # Store the categories for each column to handle new categories better
                                    categories = {}
                                    for col in onehot_columns:
                                        categories[col] = df_converted[col].unique().tolist()
                                        
                                    # Store in session state for future use
                                    if 'categorical_encoders' not in st.session_state:
                                        st.session_state['categorical_encoders'] = {}
                                    
                                    st.session_state['categorical_encoders']['onehot_categories'] = categories
                                    
                                # Create dummies and join with dataframe
                                dummies = pd.get_dummies(df_converted[onehot_columns], prefix_sep='__')
                                df_converted = df_converted.drop(columns=onehot_columns)
                                df_converted = pd.concat([df_converted, dummies], axis=1)
                                
                                st.success(f"✅ One-hot encoded {len(onehot_columns)} columns, creating {dummies.shape[1]} new columns.")
                            
                            # Process columns for label encoding
                            if label_columns:
                                # Create and store label encoders
                                if 'categorical_encoders' not in st.session_state:
                                    st.session_state['categorical_encoders'] = {}
                                
                                if 'label_encoders' not in st.session_state['categorical_encoders']:
                                    st.session_state['categorical_encoders']['label_encoders'] = {}
                                
                                for col in label_columns:
                                    le = LabelEncoder()
                                    df_converted[f"{col}_encoded"] = le.fit_transform(df_converted[col])
                                    
                                    # Store the encoder
                                    st.session_state['categorical_encoders']['label_encoders'][col] = {
                                        'encoder': le,
                                        'classes': le.classes_.tolist()
                                    }
                                    
                                    # Option to keep or drop original columns
                                    if handle_new_cats == "Use robust encoding (recommended)":
                                        # Keep original for better handling of new categories
                                        pass
                                    else:
                                        # Remove original columns
                                        df_converted = df_converted.drop(columns=[col])
                                
                                st.success(f"✅ Label encoded {len(label_columns)} columns.")
                            
                            # Store the result
                            st.session_state['df_converted'] = df_converted
                            
                            # Show download button
                            st.download_button(
                                "📥 Download Converted Data as CSV", 
                                df_converted.to_csv(index=False), 
                                file_name="converted_data.csv",
                                mime="text/csv"
                            )
                            
                            # Offer option to replace original dataframe
                            if st.button("📥 Replace Original Data with Converted Version"):
                                st.session_state['df'] = df_converted.copy()
                                st.success("Original data replaced with converted version!")
                                st.rerun()
                
                # Show result if available
                if 'df_converted' in st.session_state:
                    with st.expander("Preview Converted Data", expanded=True):
                        st.dataframe(st.session_state['df_converted'].head(10), use_container_width=True)
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original Columns", f"{df.shape[1]}")
                        with col2:
                            st.metric("Converted Columns", f"{st.session_state['df_converted'].shape[1]}")
                        with col3:
                            diff = st.session_state['df_converted'].shape[1] - df.shape[1]
                            st.metric("Column Difference", f"{diff:+d}")
                        
                        # Display encoder mappings if available
                        if 'categorical_encoders' in st.session_state and 'label_encoders' in st.session_state['categorical_encoders']:
                            st.markdown("### Label Encoder Mappings")
                            
                            for col, encoder_info in st.session_state['categorical_encoders']['label_encoders'].items():
                                if len(encoder_info['classes']) <= 20:  # Only show if not too many classes
                                    st.markdown(f"**Column: {col}**")
                                    
                                    # Create mapping DataFrame
                                    mapping_df = pd.DataFrame({
                                        'Original': encoder_info['classes'],
                                        'Encoded Value': range(len(encoder_info['classes']))
                                    })
                                    
                                    st.dataframe(mapping_df, use_container_width=True)
                                else:
                                    st.markdown(f"**Column: {col}** - {len(encoder_info['classes'])} unique values (too many to display)")
            else:
                create_card(
                    "No Categorical Columns",
                    "Your dataset doesn't have any categorical columns that need conversion.",
                    icon="ℹ️"
                )
else:
    show_file_required_warning()

# Footer
create_footer() 
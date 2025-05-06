import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
import warnings

from utils import (
    load_css, 
    create_footer, 
    show_file_required_warning, 
    display_dataset_info,
    get_numeric_columns,
    get_categorical_columns
)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Page config
st.set_page_config(page_title="Decision Tree", page_icon="🌳", layout="wide")
load_css()

# Main title
st.title("Decision Tree Classifier 🌳")
st.markdown("""
Train a decision tree model to classify data. This page helps you set up, train, 
evaluate, and visualize a decision tree model for your classification problems.
""")

# Check if data is loaded
df = st.session_state.get('df', None)
if df is not None:
    # Display dataset info
    display_dataset_info()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Train Model", "Evaluate Model", "Make Predictions"])
    
    with tab1:
        st.header("Decision Tree Overview")
        
        st.markdown("""
        ### What is a Decision Tree?
        
        A Decision Tree is a flowchart-like structure where each internal node represents a feature, 
        each branch represents a decision rule, and each leaf node represents an outcome. It's one 
        of the most interpretable machine learning models.
        
        ### When to Use Decision Trees
        
        Decision trees work well when:
        - You need a model that's easy to interpret and explain
        - The relationship between features and target is potentially non-linear
        - You're dealing with categorical or numerical features
        - You want to visualize decision-making logic
        
        ### Advantages of Decision Trees
        
        - **Interpretability**: Can be visualized and easily explained
        - **Handles mixed data**: Works with both categorical and numerical features
        - **Non-parametric**: Makes no assumptions about data distribution
        - **Automatic feature selection**: Focuses on important features
        
        ### Limitations
        
        - Tendency to overfit without proper pruning
        - Can be unstable (small changes in data can lead to large changes in tree structure)
        - May create biased trees if classes are imbalanced
        - Not ideal for capturing complex relationships that require many features
        """)
        
        # Show dataset preview
        st.subheader("Dataset Preview")
        st.dataframe(df.head(5), use_container_width=True)
        
        # Check for categorical target variables
        categorical_columns = get_categorical_columns(df)
        if categorical_columns:
            st.subheader("Potential Target Variables")
            st.markdown("These categorical columns could be suitable targets for classification:")
            
            # Show potential target variables with class distribution
            for col in categorical_columns[:5]:  # Limit to first 5 to avoid cluttering the UI
                try:
                    # Calculate value counts and percentages
                    val_counts = df[col].value_counts()
                    val_percentages = df[col].value_counts(normalize=True) * 100
                    
                    # If too many categories, only show top 5
                    if len(val_counts) > 5:
                        st.markdown(f"**{col}** (showing top 5 of {len(val_counts)} categories)")
                        counts_df = pd.DataFrame({
                            'Count': val_counts[:5],
                            'Percentage': val_percentages[:5].round(2)
                        })
                        st.dataframe(counts_df, use_container_width=True)
                    else:
                        st.markdown(f"**{col}**")
                        counts_df = pd.DataFrame({
                            'Count': val_counts,
                            'Percentage': val_percentages.round(2)
                        })
                        st.dataframe(counts_df, use_container_width=True)
                    
                    # Create bar chart for class distribution
                    fig = px.bar(
                        x=val_counts.index[:10],  # Limit to first 10 categories for visualization
                        y=val_counts.values[:10],
                        labels={'x': 'Category', 'y': 'Count'},
                        title=f"Class Distribution for {col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not analyze column {col}: {str(e)}")
        else:
            st.info("No categorical columns detected in this dataset. To use a decision tree classifier, you need a categorical target variable.")
            
    with tab2:
        st.header("Train Decision Tree Model")
        st.info("The Train Model tab will be implemented in the next update. It will include feature selection, hyperparameter tuning, and model training.")
    
    with tab3:
        st.header("Evaluate Model")
        st.info("The Evaluate Model tab will be implemented in the next update. It will include performance metrics, confusion matrix, and feature importance visualization.")
    
    with tab4:
        st.header("Make Predictions")
        st.info("The Make Predictions tab will be implemented in the next update. It will allow you to make predictions on new data.")
        
else:
    show_file_required_warning()

# Footer
create_footer() 
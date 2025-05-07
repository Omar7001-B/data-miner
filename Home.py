import streamlit as st
import os
from utils import load_css, create_footer, create_card

# Initialize theme in session state if not present
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Configure page with current theme
st.set_page_config(
    page_title="DataMiner",
    page_icon="assets/logo.png",
    layout="wide"
)

# Load custom CSS
load_css()

# Center container for logo and header
container = st.container()
with container:
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        st.markdown('''
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;">
        ''', unsafe_allow_html=True)
        st.image("assets/logo.png", width=200)
        st.markdown('''
            <h1 style="margin: 1rem 0;">Welcome to DataMiner!</h1>
            <h3 style="margin: 0.5rem 0;">Your Data Analysis Companion</h3>
            <p>Explore, visualize, and transform your data effortlessly with DataMiner.</p>
        </div>
        ''', unsafe_allow_html=True)

# Create tabs for detailed information
main_tabs = st.tabs([
    "Data Upload", "Profiling", "Summary Statistics", "Missing Values",
    "Data Cleaning", "Transformation", "Machine Learning", "Model Evaluation", "Predictions", "Visualization Tools"
])

with main_tabs[0]:
    st.markdown("### Data Upload 📤")
    st.markdown("""
    Upload and preview your data with ease:
    - Support for CSV, Excel, and JSON files
    - Interactive data preview with pagination
    - Sample datasets available (including clean and dirty data for testing)
    - Smart data type detection
    """)
    # Example datasets
    example_datasets = {
        "Iris (clean, classification)": "datasets/iris_clean.csv",
        "Titanic (dirty, missing/duplicates, classification)": "datasets/titanic_dirty.csv",
        "Synthetic (dirty, missing/duplicates/outliers, regression/classification)": "datasets/synthetic_dirty.csv"
    }
    st.markdown("#### Try an Example Dataset:")
    selected_example = st.selectbox("Choose an example dataset to load:", list(example_datasets.keys()), key="example_dataset_select")
    if st.button("Load Example Dataset", key="load_example_btn"):
        import pandas as pd
        df = pd.read_csv(example_datasets[selected_example])
        st.session_state.df = df
        st.success(f"Loaded {selected_example}!")
        st.dataframe(df.head(), use_container_width=True)
    if st.button("Start Uploading 📤", key="upload_btn"):
        st.session_state.runpage = "1_Data_Upload"
        st.rerun()

with main_tabs[1]:
    st.markdown("### Data Profiling 🔍")
    st.markdown("""
    Get detailed insights about your data:
    - Quick overview of data structure
    - Column type analysis
    - Missing value detection
    - Data quality assessment
    - Distribution visualizations
    """)
    if st.button("Profile Your Data 🔍", key="profile_btn"):
        st.session_state.runpage = "2_Profiling"
        st.rerun()

with main_tabs[2]:
    st.markdown("### Summary Statistics 📊")
    st.markdown("""
    Comprehensive statistical analysis:
    - Descriptive statistics for numeric data
    - Categorical data analysis
    - Correlation analysis
    - Distribution plots
    - Percentile calculations
    - Advanced statistical metrics
    """)
    if st.button("View Statistics 📊", key="stats_btn"):
        st.session_state.runpage = "3_Summary_Statistics"
        st.rerun()

with main_tabs[3]:
    st.markdown("### Missing Value Handling ❓")
    st.markdown("""
    Advanced missing data tools:
    - Visual missing value detection
    - Multiple imputation methods
    - Smart handling strategies
    - Before/after comparisons
    - Quality metrics
    - Automated cleaning options
    """)
    if st.button("Handle Missing Values ❓", key="missing_btn"):
        st.session_state.runpage = "4_Missing_Value_Handling"
        st.rerun()

with main_tabs[4]:
    st.markdown("### Data Cleaning 🧹")
    st.markdown("""
    Clean your data efficiently:
    - Remove duplicates
    - Handle missing values
    - Convert categorical variables
    - Visualize missing data patterns
    """)
    if st.button("Go to Data Cleaning 🧹", key="cleaning_btn"):
        st.session_state.runpage = "5_Duplicate_Entries"
        st.rerun()

with main_tabs[5]:
    st.markdown("### Data Transformation 🔄")
    st.markdown("""
    Transform your data for modeling:
    - Min-Max scaling
    - Z-score normalization
    - Decimal scaling
    - Feature selection interface
    """)
    if st.button("Go to Transformation 🔄", key="transform_btn"):
        st.session_state.runpage = "7_Data_Scaling"
        st.rerun()

with main_tabs[6]:
    st.markdown("### Machine Learning 🤖")
    st.markdown("""
    Train and evaluate models:
    - Classification: Logistic Regression, Decision Tree, KNN
    - Regression: Linear Regression
    - Interactive model training interface
    - Hyperparameter tuning
    """)
    if st.button("Go to Machine Learning 🤖", key="ml_btn"):
        st.session_state.runpage = "9_Logistic_Regression"
        st.rerun()

with main_tabs[7]:
    st.markdown("### Model Evaluation & Visualization 📈")
    st.markdown("""
    Evaluate and visualize your models:
    - Classification and regression metrics
    - Confusion matrix, ROC, PR curves
    - Residual analysis, Q-Q plots
    - Custom evaluation and visualizations
    - Model comparison and feature importance
    - Learning curves
    """)
    if st.button("Go to Model Evaluation 📈", key="eval_btn"):
        st.session_state.runpage = "13_Model_Evaluation"
        st.rerun()

with main_tabs[8]:
    st.markdown("### Make Predictions 🔮")
    st.markdown("""
    Use trained models to make predictions:
    - Single and batch predictions
    - File upload for predictions
    - Download results
    - Visualize prediction distributions
    """)
    if st.button("Go to Predictions 🔮", key="pred_btn"):
        st.session_state.runpage = "14_Make_Predictions"
        st.rerun()

with main_tabs[9]:
    st.markdown("### Advanced Visualization Tools 📊")
    st.markdown("""
    Explore advanced visualizations:
    - Model comparison (radar, bar charts)
    - Feature importance and correlation
    - Learning curves and interpretation
    - Custom data visualizations
    """)
    if st.button("Go to Visualization Tools 📊", key="viz_btn"):
        st.session_state.runpage = "15_Model_Visualization"
        st.rerun()

# Quick action cards
col1, col2 = st.columns(2)

with col1:
    card1 = st.container()
    with card1:
        st.markdown("#### 📥 New to DataMiner?")
        st.markdown("Ready to explore your data? Start by uploading your dataset or try our example datasets. We've made the process simple and intuitive for everyone!")
        if st.button("Upload Data 📤", key="quick_upload"):
            st.session_state.runpage = "1_Data_Upload"
            st.rerun()

with col2:
    card2 = st.container()
    with card2:
        st.markdown("#### 💻 Source Code")
        st.markdown("Interested in how it works? Check out the source code built by **Omar Abbas**. Whether you want to explore, star, or contribute to the project - you're welcome to join in!")
        if st.button("Visit GitHub Repository 🔗", key="github_link"):
            st.markdown("[Click here to visit the repository](https://github.com/Omar7001-B/data-miner)")

# Footer
create_footer() 
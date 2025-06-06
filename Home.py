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
    - Smart data type detection
    """)
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

# Detailed Testing Scenarios Table
st.markdown("""
---

## 🧪 Detailed Testing Scenarios

Below is a reference table for testing each page of DataMiner, including which columns to use and what output to expect:

| Page                        | Dataset         | Short Story & Steps                                                                                                   | Columns to Use (Target/Features)                | Expected Output/Result                                  |
|-----------------------------|-----------------|----------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|--------------------------------------------------------|
| 1_Data_Upload.py            | Iris            | Upload iris.csv. Preview the data and check column names.                                                            | All columns                                     | Data preview, column names, row count                  |
| 2_Profiling.py              | Titanic         | Upload titanic.csv. Profile the data: see missing values, distributions, and types.                                  | All columns                                     | Summary stats, missing value report, histograms        |
| 3_Summary_Statistics.py     | Wine Quality    | Upload winequality-red.csv. View means, medians, and ranges for each property.                                       | All columns                                     | Table of stats, boxplots, correlation matrix           |
| 4_Missing_Value_Handling.py | Titanic         | Upload titanic.csv. Impute missing "Age", drop "Cabin".                                                              | Age, Cabin                                      | Imputed Age, Cabin removed, before/after comparison    |
| 5_Duplicate_Entries.py      | Titanic         | Upload titanic.csv (with duplicates). Remove duplicates and see row count change.                                    | All columns                                     | Duplicate count, cleaned data, row count difference    |
| 6_Categorical_Conversion.py | Titanic         | Upload titanic.csv. Convert "Sex" and "Embarked" to numeric/categorical codes.                                       | Sex, Embarked                                   | Encoded columns, mapping table                        |
| 7_Data_Scaling.py           | Wine Quality    | Upload winequality-red.csv. Scale "alcohol" and "sulphates" to 0-1.                                                  | alcohol, sulphates                              | Scaled columns, min/max values                        |
| 8_Feature_Selection.py      | Breast Cancer   | Upload breast_cancer.csv. Select top 5 features for predicting "target".                                             | All features, target                            | Feature ranking, selected features                    |
| 9_Logistic_Regression.py    | Titanic         | Upload titanic.csv. Target: "Survived", Features: "Pclass", "Sex", "Age", etc. Train and evaluate.                  | Survived (target), Pclass, Sex, Age, etc.       | Accuracy, confusion matrix, coefficients              |
| 10_Decision_Tree.py         | Iris            | Upload iris.csv. Target: "species", Features: all others. Train and visualize tree.                                  | species (target), sepal/petal features          | Tree diagram, feature importance, accuracy            |
| 11_KNN_Classifier.py        | Iris            | Upload iris.csv. Target: "species", Features: all others. Try different K values.                                    | species (target), sepal/petal features          | Accuracy for each K, confusion matrix                 |
| 12_Linear_Regression.py     | Boston Housing  | Upload BostonHousing.csv. Target: "MEDV", Features: all others. Train and evaluate.                                  | MEDV (target), all other columns                | R², RMSE, scatter plot, residuals                     |
| 13_Model_Evaluation.py      | Titanic         | Upload CSV with "Survived" (actual) and "Predicted" columns. View metrics and confusion matrix.                      | Survived, Predicted                             | Accuracy, precision, recall, confusion matrix         |
| 14_Make_Predictions.py      | Iris            | Upload new iris samples. Use trained model to predict "species".                                                     | sepal/petal features                            | Predicted species, probability table                  |
| 15_Model_Visualization.py   | Iris            | Upload iris.csv, train models. Compare models, see feature importance, and view learning curves.                     | species (target), sepal/petal features          | Model comparison charts, feature importance, curves   |

---

**Tip:** Use the columns listed for each test. Outputs include tables, plots, and metrics to help you verify each feature!
""") 
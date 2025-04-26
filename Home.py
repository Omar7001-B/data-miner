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
tab1, tab2, tab3, tab4 = st.tabs(["Data Upload", "Profiling", "Summary Statistics", "Missing Values"])

with tab1:
    st.markdown("### Data Upload 📤")
    st.markdown("""
    Upload and preview your data with ease:
    - Support for CSV, Excel, and JSON files
    - Interactive data preview with pagination
    - Sample datasets available
    - Smart data type detection
    """)
    if st.button("Start Uploading 📤", key="upload_btn"):
        st.session_state.runpage = "1_Data_Upload"
        st.rerun()

with tab2:
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

with tab3:
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

with tab4:
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
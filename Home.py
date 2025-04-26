import streamlit as st
import os
from utils import load_css, create_footer, create_card

# Initialize theme in session state if not present
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Configure page with current theme
st.set_page_config(
    page_title="DataMiner Home",
    page_icon="🏠",
    layout="wide"
)

# Load custom CSS
load_css()

# Header with logo image (if exists)
col1, col2 = st.columns([1, 3])
with col1:
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", width=120)
with col2:
    st.title("Welcome to DataMiner! 🏠")

# Main content with engaging intro
st.markdown("""
### 🚀 Unlock the Power of Your Data!

DataMiner helps you explore, visualize, and transform your datasets with just a few clicks.
No coding required - our interactive tools make data analysis accessible to everyone.
""")

# Quick start guide in a colored container
st.markdown("""
<div class="quick-start-guide">
<h3>🏁 Quick Start Guide</h3>
<ol>
<li><b>Upload your data</b> - Go to the <a href="Data_Upload" target="_self">Data Upload</a> page and import your CSV, Excel, or JSON file</li>
<li><b>Explore your dataset</b> - Use the <a href="Profiling" target="_self">Profiling</a> page to understand your data's structure</li>
<li><b>Analyze statistics</b> - Check the <a href="Summary_Statistics" target="_self">Summary Statistics</a> page for key metrics</li>
<li><b>Handle missing values</b> - Clean your data with the <a href="Missing_Value_Handling" target="_self">Missing Value Handling</a> tools</li>
</ol>
</div>
""", unsafe_allow_html=True)

# Features section using tabs
st.subheader("📊 Features")
tab1, tab2, tab3, tab4 = st.tabs(["Data Upload", "Profiling", "Summary Statistics", "Missing Value Handling"])

with tab1:
    create_card(
        "Data Upload",
        """
        - Support for CSV, Excel, and JSON files
        - Paginated data preview
        - Interactive data browser
        """,
        icon="📤"
    )
    
with tab2:
    create_card(
        "Profiling",
        """
        - Quick shape and column overview
        - Detailed data type analysis
        - Instant data structure insights
        """,
        icon="🔍"
    )
    
with tab3:
    create_card(
        "Summary Statistics",
        """
        - Comprehensive numeric statistics
        - Categorical data analysis
        - Detailed explanations of all metrics
        """,
        icon="📊"
    )
    
with tab4:
    create_card(
        "Missing Value Handling",
        """
        - Visual missing value detection
        - Multiple imputation methods
        - Before/after comparison views
        - Download cleaned datasets
        """,
        icon="❓"
    )

# Footer
create_footer() 
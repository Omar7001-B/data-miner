import streamlit as st
import pandas as pd
import time
from utils import load_css, create_footer, display_dataset_info

st.set_page_config(page_title="Data Upload", page_icon="📤", layout="wide")
load_css()

st.title("Data Upload 📤")
st.markdown("Upload your dataset to begin analysis. We support CSV, Excel, and JSON formats.")

# If a DataFrame is already in session_state, use it
if 'df' in st.session_state:
    df = st.session_state['df']
    
    st.success("✅ Data successfully loaded!")
    display_dataset_info()
    
    with st.expander("Preview Data", expanded=True):
        # Add filter options
        col1, col2 = st.columns([1, 3])
        with col1:
            rows_per_page = st.selectbox("Rows per page", [10, 25, 50, 100], index=0)
        with col2:
            search_term = st.text_input("Search in data (case sensitive)", "")
        
        # Apply filters
        if search_term:
            filtered_df = df[df.astype(str).apply(lambda row: row.str.contains(search_term).any(), axis=1)]
            st.write(f"Found {len(filtered_df)} rows containing '{search_term}'")
        else:
            filtered_df = df
            
        # Pagination
        total_rows = filtered_df.shape[0]
        total_pages = (total_rows - 1) // rows_per_page + 1
        
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("◀️ Previous"):
                st.session_state["page"] = max(1, st.session_state.get("page", 1) - 1)
        with col2:
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=st.session_state.get("page", 1))
            st.session_state["page"] = page
        with col3:
            if st.button("Next ▶️"):
                st.session_state["page"] = min(total_pages, st.session_state.get("page", 1) + 1)
        
        start = (page - 1) * rows_per_page
        end = min(start + rows_per_page, len(filtered_df))
        
        st.dataframe(filtered_df.iloc[start:end], use_container_width=True)
        st.caption(f"Showing rows {start+1} to {end} of {total_rows}")
    
    # Option to upload a different file
    if st.button("Upload a different file"):
        del st.session_state['df']
        st.rerun()
        
else:
    # File uploader with options
    upload_tab, example_tab = st.tabs(["Upload Your File", "Use Example Dataset"])
    
    with upload_tab:
        uploaded_file = st.file_uploader(
            "Choose a data file",
            type=["csv", "xlsx", "xls", "json"],
            help="Upload a CSV, Excel, or JSON file to get started"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            has_header = st.checkbox("File has header row", value=True)
        with col2:
            delimiter = st.selectbox("CSV Delimiter", [",", ";", "tab", "|", ":", " "], disabled=(uploaded_file is None))
        
        if uploaded_file:
            file_type = uploaded_file.name.split(".")[-1].lower()
            
            with st.spinner("Loading data..."):
                try:
                    if file_type in ["csv"]:
                        sep = "\t" if delimiter == "tab" else delimiter
                        df = pd.read_csv(uploaded_file, header=0 if has_header else None, sep=sep)
                    elif file_type in ["xlsx", "xls"]:
                        df = pd.read_excel(uploaded_file, header=0 if has_header else None)
                    elif file_type == "json":
                        df = pd.read_json(uploaded_file)
                    
                    time.sleep(0.5)  # Simulate processing for better UX
                    st.session_state['df'] = df
                    st.success(f"✅ Successfully loaded {uploaded_file.name}!")
                    st.toast(f"Loaded {uploaded_file.name} with {df.shape[0]} rows and {df.shape[1]} columns")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading file: {e}")
            
    with example_tab:
        example_option = st.selectbox(
            "Choose an example dataset",
            ["Iris Flower Dataset", "Titanic Passengers", "Wine Quality"]
        )
        
        if st.button("Load Example"):
            with st.spinner("Loading example dataset..."):
                try:
                    if example_option == "Iris Flower Dataset":
                        df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
                    elif example_option == "Titanic Passengers":
                        df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
                    elif example_option == "Wine Quality":
                        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")
                    
                    time.sleep(0.5)  # Simulate processing for better UX
                    st.session_state['df'] = df
                    st.success(f"✅ Successfully loaded {example_option}!")
                    st.toast(f"Loaded {example_option} with {df.shape[0]} rows and {df.shape[1]} columns")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading example dataset: {e}")

    # Display helpful tips
    st.markdown("""
    <div style="background-color:#262730; padding:15px; border-radius:5px; border-left:4px solid #FF4B4B;">
      <h4>📋 Data Loading Tips</h4>
      <ul>
        <li>Make sure your CSV or Excel file has proper formatting</li>
        <li>For large files, loading might take a few moments</li>
        <li>If you encounter errors, check your file for special characters or inconsistent data types</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer        
create_footer() 
import streamlit as st
import pandas as pd
import time
from utils import load_css, create_footer, display_dataset_info
import os

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
            # Ensure the default value is within bounds
            current_page = st.session_state.get("page", 1)
            if current_page < 1:
                current_page = 1
            if current_page > total_pages:
                current_page = total_pages
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=current_page)
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
    upload_tab, example_tab, local_tab = st.tabs(["Upload Your File", "Use Example Dataset", "Use Local Dataset"])
    
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
            [
                "Iris Flower Dataset (sklearn)",
                "Titanic Passengers (seaborn)",
                "Wine Quality (UCI)",
                "Breast Cancer (sklearn)",
                "Boston Housing (sklearn)",
                "Digits (sklearn)",
                "Synthetic Dirty (code snippet)"
            ]
        )
        
        if st.button("Load Example"):
            with st.spinner("Loading example dataset..."):
                try:
                    if example_option == "Iris Flower Dataset (sklearn)":
                        from sklearn.datasets import load_iris
                        data = load_iris(as_frame=True)
                        df = data.frame
                    elif example_option == "Titanic Passengers (seaborn)":
                        import seaborn as sns
                        df = sns.load_dataset('titanic')
                    elif example_option == "Wine Quality (UCI)":
                        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")
                    elif example_option == "Breast Cancer (sklearn)":
                        from sklearn.datasets import load_breast_cancer
                        data = load_breast_cancer(as_frame=True)
                        df = data.frame
                    elif example_option == "Boston Housing (sklearn)":
                        from sklearn.datasets import load_boston
                        data = load_boston()
                        import numpy as np
                        df = pd.DataFrame(data.data, columns=data.feature_names)
                        df['target'] = data.target
                    elif example_option == "Digits (sklearn)":
                        from sklearn.datasets import load_digits
                        data = load_digits(as_frame=True)
                        df = data.frame
                    elif example_option == "Synthetic Dirty (code snippet)":
                        st.info("""
                        To generate a synthetic dirty dataset, run this code in your Python environment:
                        ```python
                        import pandas as pd
                        import numpy as np
                        np.random.seed(42)
                        df = pd.DataFrame({
                            'id': range(1, 21),
                            'feature1': [10,12,10,15,13,1000,14,13,12,np.nan,11,10,15,14,13,12,10,1000,11,10],
                            'feature2': [5.2,np.nan,5.2,8.1,7.5,8.0,7.8,7.5,6.9,7.0,5.5,5.2,8.1,7.8,7.5,6.9,5.2,8.0,5.5,5.2],
                            'feature3': ['cat','dog','cat','cat','dog','cat',np.nan,'dog','cat','dog','cat','cat','cat','dog','dog','cat','cat','cat','cat','cat'],
                            'target': [100,110,100,200,150,10000,160,150,120,130,105,100,200,160,150,120,100,10000,105,100]
                        })
                        df.to_csv('synthetic_dirty.csv', index=False)
                        ```
                        Then upload 'synthetic_dirty.csv' here.
                        """)
                        raise Exception("Synthetic dataset must be generated locally.")
                    time.sleep(0.5)  # Simulate processing for better UX
                    st.session_state['df'] = df
                    st.success(f"✅ Successfully loaded {example_option}!")
                    st.toast(f"Loaded {example_option} with {df.shape[0]} rows and {df.shape[1]} columns")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading example dataset: {e}")

    with local_tab:
        local_files = [f for f in os.listdir('datasets') if f.endswith('.csv')]
        if not local_files:
            st.info("No CSV files found in the /datasets folder.")
        else:
            selected_local = st.selectbox("Select a local dataset to load:", local_files)
            if st.button("Load Local Dataset"):
                path = os.path.join('datasets', selected_local)
                try:
                    df = pd.read_csv(path)
                    st.session_state['df'] = df
                    st.success(f"✅ Successfully loaded {selected_local}!")
                    st.dataframe(df.head(), use_container_width=True)
                    st.toast(f"Loaded {selected_local} with {df.shape[0]} rows and {df.shape[1]} columns")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading local dataset: {e}")

    # Display helpful tips
    st.markdown("""
    <div class="stTips">
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
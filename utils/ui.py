"""
UI utility functions for DataMiner

This module contains helper functions for UI components across the application.
"""

import streamlit as st
import os

def create_footer():
    """Display a consistent footer across all pages."""
    st.markdown("""
    <div class="footer">
        Made with ❤️ by Omar Abbas
    </div>
    """, unsafe_allow_html=True)
    
def load_css():
    """Load custom CSS if it exists."""
    if os.path.exists("assets/style.css"):
        with open("assets/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            
def show_file_required_warning():
    """Display a standardized warning when no file is uploaded."""
    st.warning("""
    ⚠️ No data found! Please upload a file on the Data Upload page first.
    
    Go to [Data Upload](/Data_Upload) to get started.
    """)
    
def create_card(title, content, icon="📊", color="#262730"):
    """
    Create a styled card with title, content and an icon.
    
    Parameters:
    -----------
    title : str
        The card title
    content : str
        The content text (can include markdown)
    icon : str
        An emoji or icon to display
    color : str
        Border color for the card
    """
    st.markdown(f"""
    <div style="
        background-color:{color}; 
        padding:15px; 
        border-radius:5px; 
        margin-bottom:15px;
        border-left: 4px solid #FF4B4B;
    ">
        <h4>{icon} {title}</h4>
        <div>{content}</div>
    </div>
    """, unsafe_allow_html=True)
    
def display_dataset_info():
    """Display standardized dataset information across pages."""
    if 'df' in st.session_state:
        df = st.session_state['df']
        cols = st.columns(4)
        with cols[0]:
            st.metric("Rows", f"{df.shape[0]:,}")
        with cols[1]:
            st.metric("Columns", f"{df.shape[1]:,}")
        with cols[2]:
            missing = df.isna().sum().sum()
            st.metric("Missing Values", f"{missing:,}")
        with cols[3]:
            memory = df.memory_usage(deep=True).sum()
            if memory < 1024:
                memory_str = f"{memory} bytes"
            elif memory < 1024**2:
                memory_str = f"{memory/1024:.1f} KB"
            else:
                memory_str = f"{memory/1024**2:.1f} MB"
            st.metric("Memory Usage", memory_str) 
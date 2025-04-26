"""
Data utility functions for DataMiner

This module contains helper functions for data processing and transformation.
"""

import pandas as pd
import numpy as np

def get_numeric_columns(df):
    """Return a list of numeric columns in the DataFrame."""
    return df.select_dtypes(include=['number']).columns.tolist()

def get_categorical_columns(df):
    """Return a list of categorical columns in the DataFrame."""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def get_datetime_columns(df):
    """Return a list of datetime columns in the DataFrame."""
    return df.select_dtypes(include=['datetime']).columns.tolist()

def check_missing_values(df):
    """
    Calculate missing values in a DataFrame.
    
    Returns:
    --------
    pd.DataFrame
        A DataFrame with missing count and percentage for each column
    """
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing %': missing_percent
    })
    missing_df = missing_df.sort_values('Missing Count', ascending=False)
    return missing_df

def generate_summary_stats(df, columns=None):
    """
    Generate enhanced summary statistics for specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to analyze
    columns : list, optional
        List of column names to analyze. If None, uses all numeric columns.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with detailed statistics
    """
    if columns is None:
        columns = get_numeric_columns(df)
    
    # Get basic statistics
    stats_df = df[columns].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
    
    # Add more statistics
    stats_df.loc['skew'] = df[columns].skew()
    stats_df.loc['kurtosis'] = df[columns].kurtosis()
    stats_df.loc['missing'] = df[columns].isnull().sum()
    stats_df.loc['missing %'] = df[columns].isnull().sum() / len(df) * 100
    
    return stats_df 
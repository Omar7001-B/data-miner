"""
Visualization utility functions for DataMiner

This module contains helper functions for creating plots and visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st

def plot_missing_heatmap(df, top_n=20, max_rows=100):
    """
    Create a heatmap of missing values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to visualize
    top_n : int, optional
        Number of columns with most missing values to show
    max_rows : int, optional
        Maximum number of rows to display
        
    Returns:
    --------
    fig : matplotlib Figure
        The figure object containing the visualization
    """
    # Get columns with most missing values
    missing_count = df.isnull().sum()
    top_missing_cols = missing_count.sort_values(ascending=False).head(top_n).index
    
    # Prepare heatmap data
    heatmap_data = df[top_missing_cols].isnull().iloc[:max_rows]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        heatmap_data,
        cbar=True,
        yticklabels=False,
        cmap="YlGnBu",
        ax=ax,
        linewidths=0.5,
        linecolor='white'
    )
    ax.set_title(f"Missing Data Heatmap (Top {top_n} Columns, First {max_rows} Rows)")
    ax.set_xlabel("Columns")
    ax.set_ylabel(f"Rows (first {max_rows})")
    plt.xticks(rotation=90, ha='center')
    
    return fig

def plot_correlation_heatmap(df, columns=None, method='pearson'):
    """
    Create a correlation heatmap for numeric columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to visualize
    columns : list, optional
        List of columns to use. If None, uses all numeric columns.
    method : str, optional
        Correlation method: 'pearson', 'kendall', or 'spearman'
        
    Returns:
    --------
    fig : matplotlib Figure
        The figure object containing the visualization
    """
    # Select columns if not provided
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns
    
    # Calculate correlation
    corr_matrix = df[columns].corr(method=method).round(2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, len(columns) * 0.8), max(8, len(columns) * 0.5)))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Draw heatmap
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True, 
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1, 
        center=0,
        square=True, 
        linewidths=.5, 
        fmt=".2f",
        ax=ax
    )
    ax.set_title(f"{method.capitalize()} Correlation Matrix")
    plt.tight_layout()
    
    return fig

def plot_distribution(df, column):
    """
    Plot the distribution of a column based on its data type.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to visualize
    column : str
        The column name to plot
        
    Returns:
    --------
    fig : matplotlib Figure
        The figure object containing the visualization
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if pd.api.types.is_numeric_dtype(df[column]):
        # For numeric data, use histogram with KDE
        sns.histplot(df[column].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {column}")
    else:
        # For categorical data, use a bar chart of counts
        value_counts = df[column].value_counts().head(20)  # Limit to top 20 categories
        
        # If there are too many categories, make the plot taller
        plot_height = min(10, max(5, len(value_counts) * 0.25))
        fig, ax = plt.subplots(figsize=(10, plot_height))
        
        # Plot horizontal barplot for better label display
        bars = sns.barplot(y=value_counts.index, x=value_counts.values, ax=ax)
        
        # Add percentage labels
        total = value_counts.sum()
        for i, p in enumerate(bars.patches):
            percentage = (value_counts.values[i] / total) * 100
            ax.annotate(f"{percentage:.1f}%", 
                       (p.get_width() + 0.1, p.get_y() + p.get_height()/2),
                       ha='left', va='center')
        
        ax.set_title(f"Distribution of {column} (Top 20 values)")
        ax.set_xlabel("Count")
    
    plt.tight_layout()
    return fig

def plot_scatter(df, x_col, y_col, color_col=None, size_col=None):
    """
    Create a scatter plot of two variables with optional color and size dimensions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to visualize
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    color_col : str, optional
        Column name to use for point colors
    size_col : str, optional
        Column name to use for point sizes
        
    Returns:
    --------
    fig : matplotlib Figure
        The figure object containing the visualization
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Handle optional parameters
    plot_kwargs = {}
    if color_col is not None:
        plot_kwargs['hue'] = color_col
    if size_col is not None:
        plot_kwargs['size'] = size_col
    
    # Create the scatter plot
    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, **plot_kwargs)
    
    # Add title and adjust layout
    ax.set_title(f"{y_col} vs {x_col}")
    plt.tight_layout()
    
    return fig 
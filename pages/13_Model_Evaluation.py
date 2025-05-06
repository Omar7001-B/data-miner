import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    mean_squared_error, r2_score, mean_absolute_error,
    mean_absolute_percentage_error, explained_variance_score
)
import io
import base64
from datetime import datetime
import warnings

from utils import (
    load_css, 
    create_footer, 
    show_file_required_warning, 
    display_dataset_info
)

# Suppress warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(page_title="Model Evaluation", page_icon="📊", layout="wide")
load_css()

# Main title
st.title("Model Evaluation 📊")
st.markdown("""
This page provides comprehensive evaluation metrics for your machine learning models. 
Upload model predictions and actual values to evaluate performance with various metrics and visualizations.
""")

# Check if data is loaded
df = st.session_state.get('df', None)
if df is not None:
    # Display dataset info
    display_dataset_info()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Classification Metrics", "Regression Metrics", "Custom Evaluation"])
    
    with tab1:
        st.header("Classification Metrics Calculator")
        
        st.markdown("""
        Evaluate classification model performance with standard metrics and visualizations.
        You can either:
        1. Upload predictions from any model
        2. Choose from your previously trained models
        """)
        
        # Data source options
        data_source = st.radio(
            "Select data source:",
            options=["Upload predictions", "Use trained model"],
            horizontal=True
        )
        
        if data_source == "Upload predictions":
            st.subheader("Upload Predictions")
            
            # File uploader for predictions
            uploaded_file = st.file_uploader(
                "Upload CSV file with actual and predicted values", 
                type=["csv"], 
                key="classification_upload"
            )
            
            if uploaded_file is not None:
                try:
                    # Read the data
                    pred_df = pd.read_csv(uploaded_file)
                    
                    # Display preview
                    st.subheader("Data Preview")
                    st.dataframe(pred_df.head(), use_container_width=True)
                    
                    # Check if required columns exist
                    st.subheader("Column Selection")
                    st.markdown("Please select the columns containing actual and predicted values.")
                    
                    # Get column names
                    columns = pred_df.columns.tolist()
                    
                    # Select actual and predicted columns
                    actual_col = st.selectbox("Select column with actual values:", options=columns)
                    pred_col = st.selectbox("Select column with predicted values:", options=columns, index=min(1, len(columns)-1))
                    
                    # Check if the selected columns are valid
                    if actual_col and pred_col:
                        # Extract actual and predicted values
                        y_true = pred_df[actual_col]
                        y_pred = pred_df[pred_col]
                        
                        # Basic validation
                        if len(y_true) != len(y_pred):
                            st.error("Error: Actual and predicted values must have the same length.")
                        else:
                            # Check if binary or multiclass
                            unique_classes = np.unique(np.concatenate([y_true.unique(), y_pred.unique()]))
                            is_binary = len(unique_classes) <= 2
                            
                            # Display classification type
                            if is_binary:
                                st.info("Binary classification detected.")
                            else:
                                st.info(f"Multiclass classification detected with {len(unique_classes)} classes.")
                            
                            # Class label format
                            class_format = st.radio(
                                "Select class label format:",
                                options=["Numeric (0, 1, 2, ...)", "String ('yes', 'no', ...)"],
                                horizontal=True
                            )
                            
                            # Convert to appropriate format if needed
                            if class_format == "Numeric (0, 1, 2, ...)":
                                try:
                                    # Try to convert to numeric
                                    y_true = pd.to_numeric(y_true)
                                    y_pred = pd.to_numeric(y_pred)
                                except Exception as e:
                                    st.error(f"Error converting to numeric: {str(e)}")
                            
                            # Continue with evaluation
                            st.subheader("Evaluate Model")
                            
                            if st.button("Calculate Metrics"):
                                # Calculate and display metrics
                                display_classification_metrics(y_true, y_pred, is_binary)
                
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    st.info("Please ensure your CSV file is properly formatted with columns for actual and predicted values.")
        
        else:  # Use trained model
            st.subheader("Use Trained Model")
            
            # Check for trained models in session state
            models = []
            if 'lr_trained_model' in st.session_state and st.session_state.lr_trained_model is not None:
                models.append("Logistic Regression")
            if 'dt_trained_model' in st.session_state and st.session_state.dt_trained_model is not None:
                models.append("Decision Tree")
            if 'knn_trained_model' in st.session_state and st.session_state.knn_trained_model is not None:
                models.append("K-Nearest Neighbors")
            
            if not models:
                st.warning("No trained classification models found. Please train a model first.")
            else:
                # Select model
                selected_model = st.selectbox("Select trained model:", options=models)
                
                # Get model data based on selection
                if selected_model == "Logistic Regression" and 'lr_trained_model' in st.session_state:
                    y_true = st.session_state.lr_y_test
                    y_pred = st.session_state.lr_pipeline.predict(st.session_state.lr_X_test)
                    classes = st.session_state.lr_classes if 'lr_classes' in st.session_state else None
                    is_binary = len(np.unique(y_true)) <= 2
                    
                elif selected_model == "Decision Tree" and 'dt_trained_model' in st.session_state:
                    y_true = st.session_state.dt_y_test
                    y_pred = st.session_state.dt_pipeline.predict(st.session_state.dt_X_test)
                    classes = st.session_state.dt_classes if 'dt_classes' in st.session_state else None
                    is_binary = len(np.unique(y_true)) <= 2
                    
                elif selected_model == "K-Nearest Neighbors" and 'knn_trained_model' in st.session_state:
                    y_true = st.session_state.knn_y_test
                    y_pred = st.session_state.knn_pipeline.predict(st.session_state.knn_X_test)
                    classes = st.session_state.knn_classes if 'knn_classes' in st.session_state else None
                    is_binary = len(np.unique(y_true)) <= 2
                
                # Calculate and display metrics
                if 'y_true' in locals() and 'y_pred' in locals():
                    st.subheader(f"Evaluation for {selected_model}")
                    
                    # Display information about the data
                    st.info(f"Using test data with {len(y_true)} samples.")
                    
                    # Calculate and display metrics
                    display_classification_metrics(y_true, y_pred, is_binary, classes)
                else:
                    st.error("Could not retrieve model data. Please retrain the model.")


def display_classification_metrics(y_true, y_pred, is_binary, classes=None):
    """
    Calculate and display classification metrics with visualizations.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values
    y_pred : array-like
        Estimated targets as returned by a classifier
    is_binary : bool
        Whether this is a binary classification problem
    classes : array-like, optional
        List of class names
    """
    # Create tabs for different metric groups
    metric_tab1, metric_tab2, metric_tab3, metric_tab4 = st.tabs([
        "Basic Metrics", 
        "Confusion Matrix", 
        "ROC & PR Curves", 
        "Classification Report"
    ])
    
    with metric_tab1:
        st.subheader("Basic Classification Metrics")
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        if is_binary:
            precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        else:
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Display metrics in a grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
            st.caption("Proportion of correct predictions")
        
        with col2:
            st.metric("Precision", f"{precision:.4f}")
            st.caption("Correctly predicted positive cases / All predicted positive cases")
        
        with col3:
            st.metric("Recall", f"{recall:.4f}")
            st.caption("Correctly predicted positive cases / All actual positive cases")
        
        with col4:
            st.metric("F1 Score", f"{f1:.4f}")
            st.caption("Harmonic mean of precision and recall")
        
        # Advanced metrics for multiclass
        if not is_binary:
            st.subheader("Multiclass Metrics")
            
            # Calculate metrics with different averaging
            metrics_df = pd.DataFrame({
                'Metric': ['Precision', 'Recall', 'F1 Score'],
                'Macro Avg': [
                    precision_score(y_true, y_pred, average='macro', zero_division=0),
                    recall_score(y_true, y_pred, average='macro', zero_division=0),
                    f1_score(y_true, y_pred, average='macro', zero_division=0)
                ],
                'Weighted Avg': [
                    precision_score(y_true, y_pred, average='weighted', zero_division=0),
                    recall_score(y_true, y_pred, average='weighted', zero_division=0),
                    f1_score(y_true, y_pred, average='weighted', zero_division=0)
                ],
                'Micro Avg': [
                    precision_score(y_true, y_pred, average='micro', zero_division=0),
                    recall_score(y_true, y_pred, average='micro', zero_division=0),
                    f1_score(y_true, y_pred, average='micro', zero_division=0)
                ]
            })
            
            # Display advanced metrics
            st.dataframe(metrics_df.set_index('Metric').round(4), use_container_width=True)
            
            # Add explanation of averaging methods
            with st.expander("Understanding Multiclass Averaging Methods"):
                st.markdown("""
                - **Macro Average**: Simple arithmetic mean of metrics for each class. Treats all classes equally.
                - **Weighted Average**: Weighted mean of metrics for each class, weighted by support (number of instances).
                - **Micro Average**: Aggregate contributions of all classes to compute the average metric.
                """)
        
        # Interpretation guidance
        with st.expander("Interpreting Basic Metrics"):
            st.markdown("""
            ### Metric Interpretation Guide
            
            - **Accuracy**: Percentage of correct predictions. Best when classes are balanced.
            - **Precision**: Measures how many of the positive predictions were actually correct. 
              High precision means low false positive rate.
            - **Recall**: Measures how many of the actual positives were correctly predicted. 
              High recall means low false negative rate.
            - **F1 Score**: Harmonic mean of precision and recall. Provides balance between the two.
            
            ### When to focus on which metric?
            
            - **Accuracy**: Good for balanced datasets where all classes are equally important
            - **Precision**: When false positives are costly (e.g., spam detection)
            - **Recall**: When false negatives are costly (e.g., disease detection)
            - **F1 Score**: When you need balance between precision and recall
            """)
    
    with metric_tab2:
        st.subheader("Confusion Matrix")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Get class labels
        if classes is None:
            unique_classes = sorted(np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)])))
            class_labels = [str(cls) for cls in unique_classes]
        else:
            class_labels = classes
        
        # Normalization options
        normalize_option = st.radio(
            "Normalization:",
            options=["None", "By Row (Recall)", "By Column (Precision)", "All"],
            horizontal=True
        )
        
        # Normalize confusion matrix if requested
        if normalize_option == "None":
            cm_display = cm
            fmt = '.0f'
            title = "Confusion Matrix (Counts)"
        elif normalize_option == "By Row (Recall)":
            cm_sums = cm.sum(axis=1)[:, np.newaxis]
            cm_display = np.divide(cm, cm_sums, where=cm_sums!=0)
            fmt = '.2f'
            title = "Confusion Matrix (Normalized by Row)"
        elif normalize_option == "By Column (Precision)":
            cm_sums = cm.sum(axis=0)[np.newaxis, :]
            cm_display = np.divide(cm, cm_sums, where=cm_sums!=0)
            fmt = '.2f'
            title = "Confusion Matrix (Normalized by Column)"
        else:  # "All"
            cm_display = cm / cm.sum()
            fmt = '.2f'
            title = "Confusion Matrix (Normalized by Total)"
        
        # Plot confusion matrix
        fig = px.imshow(
            cm_display,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=class_labels,
            y=class_labels,
            color_continuous_scale="Blues",
            text_auto=fmt
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Predicted Class",
            yaxis_title="Actual Class",
            width=600,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation guidance
        with st.expander("Interpreting the Confusion Matrix"):
            st.markdown("""
            ### How to Read a Confusion Matrix
            
            - **Rows**: Represent the actual classes
            - **Columns**: Represent the predicted classes
            - **Diagonal elements**: Correct predictions
            - **Off-diagonal elements**: Incorrect predictions
            
            ### Normalization Options
            
            - **None**: Shows raw counts
            - **By Row**: Each row sums to 1, useful for seeing recall (true positive rate) per class
            - **By Column**: Each column sums to 1, useful for seeing precision per class
            - **All**: Shows proportion of total samples, useful for imbalanced datasets
            
            ### Key Insights from Confusion Matrix
            
            - Classes with many off-diagonal elements indicate where the model struggles
            - Pattern of misclassifications can suggest feature improvements
            - Imbalanced datasets may show good accuracy but poor performance on minority classes
            """)
    
    with metric_tab3:
        st.subheader("ROC and Precision-Recall Curves")
        
        # Check if the model provides probability estimates
        try:
            # For binary classification
            if is_binary:
                st.markdown("### ROC Curve (Receiver Operating Characteristic)")
                
                # Get binary predictions if available (assuming class 1 is positive)
                # This part needs to be customized based on the actual model and classes
                try:
                    # If y_true and y_pred are already binary (0/1), use them directly
                    if set(np.unique(y_true)) <= {0, 1} and set(np.unique(y_pred)) <= {0, 1}:
                        y_true_binary = y_true
                        y_pred_binary = y_pred
                    else:
                        # Otherwise, try to convert to binary based on class labels
                        # Let's assume the second class (index 1) is the positive class
                        positive_class = np.unique(y_true)[1] if len(np.unique(y_true)) > 1 else np.unique(y_true)[0]
                        y_true_binary = (y_true == positive_class).astype(int)
                        y_pred_binary = (y_pred == positive_class).astype(int)
                    
                    # Calculate ROC curve and ROC area
                    fpr, tpr, _ = roc_curve(y_true_binary, y_pred_binary)
                    roc_auc = auc(fpr, tpr)
                    
                    # Create ROC curve plot
                    fig = px.line(
                        x=fpr, y=tpr,
                        labels=dict(x='False Positive Rate', y='True Positive Rate'),
                        title=f'ROC Curve (AUC = {roc_auc:.4f})'
                    )
                    
                    # Add diagonal reference line (random classifier)
                    fig.add_trace(
                        go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode='lines',
                            name='Random Classifier',
                            line=dict(dash='dash', color='gray')
                        )
                    )
                    
                    # Update first trace name and color
                    fig.data[0].name = 'ROC Curve'
                    fig.data[0].line.color = 'blue'
                    
                    # Update layout
                    fig.update_layout(
                        xaxis=dict(range=[0, 1]),
                        yaxis=dict(range=[0, 1.05]),
                        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)'),
                        width=700,
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add PR curve
                    st.markdown("### Precision-Recall Curve")
                    
                    # Calculate precision-recall curve
                    precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_binary)
                    pr_auc = average_precision_score(y_true_binary, y_pred_binary)
                    
                    # Create PR curve plot
                    fig_pr = px.line(
                        x=recall, y=precision,
                        labels=dict(x='Recall', y='Precision'),
                        title=f'Precision-Recall Curve (AUC = {pr_auc:.4f})'
                    )
                    
                    # Add horizontal line for baseline
                    baseline = sum(y_true_binary) / len(y_true_binary)
                    fig_pr.add_trace(
                        go.Scatter(
                            x=[0, 1], y=[baseline, baseline],
                            mode='lines',
                            name='Baseline',
                            line=dict(dash='dash', color='gray')
                        )
                    )
                    
                    # Update first trace name and color
                    fig_pr.data[0].name = 'PR Curve'
                    fig_pr.data[0].line.color = 'green'
                    
                    # Update layout
                    fig_pr.update_layout(
                        xaxis=dict(range=[0, 1]),
                        yaxis=dict(range=[0, 1.05]),
                        legend=dict(x=0.01, y=0.01, bgcolor='rgba(255, 255, 255, 0.8)'),
                        width=700,
                        height=500
                    )
                    
                    st.plotly_chart(fig_pr, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Could not generate ROC and PR curves: {str(e)}")
                    st.info("ROC and PR curves require probability outputs or binary class labels.")
            
            else:
                # For multiclass, show class-wise ROC curves if available
                st.info("For multiclass classification, class-wise ROC curves would be shown here.")
                st.warning("This feature is still in development.")
        
        except Exception as e:
            st.warning(f"Could not generate ROC and PR curves: {str(e)}")
            st.info("ROC and PR curves require probability outputs which are not available.")
        
        # Interpretation guidance
        with st.expander("Interpreting ROC and PR Curves"):
            st.markdown("""
            ### ROC Curve
            
            The **Receiver Operating Characteristic (ROC)** curve plots the True Positive Rate (Sensitivity) against the False Positive Rate (1-Specificity) at various classification thresholds.
            
            - **AUC (Area Under Curve)**: Measures the classifier's ability to distinguish between classes
              - AUC = 0.5: No discrimination (equivalent to random guessing)
              - 0.7 ≤ AUC < 0.8: Acceptable discrimination
              - 0.8 ≤ AUC < 0.9: Excellent discrimination
              - AUC ≥ 0.9: Outstanding discrimination
            
            - **When to use**: Good for balanced datasets and when you care about ranking predictions
            
            ### Precision-Recall Curve
            
            The **Precision-Recall (PR)** curve plots Precision against Recall at various classification thresholds.
            
            - **Average Precision (AP)**: Summarizes the PR curve as the weighted mean of precisions at each threshold
            
            - **When to use**: Better than ROC for imbalanced datasets, focusing on performance of the positive class
            
            ### Choosing a Threshold
            
            These curves help you select an optimal classification threshold based on your specific needs:
            - Move toward the top-left of the ROC curve to reduce false positives
            - Move toward the top-right of the PR curve to improve recall
            """)
    
    with metric_tab4:
        st.subheader("Classification Report")
        
        # Generate classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Convert to DataFrame and format
        report_df = pd.DataFrame(report).transpose()
        
        # Drop unnecessary row for better display
        if 'accuracy' in report_df.index:
            report_df = report_df.drop('accuracy')
        
        # Round numeric values
        report_df = report_df.round(4)
        
        # Rename columns for clarity
        report_df.columns = ['Precision', 'Recall', 'F1-Score', 'Support']
        
        # Customize index names for clarity
        if 'macro avg' in report_df.index:
            report_df = report_df.rename(index={'macro avg': 'Macro Avg'})
        if 'weighted avg' in report_df.index:
            report_df = report_df.rename(index={'weighted avg': 'Weighted Avg'})
        
        # Display as table
        st.dataframe(report_df, use_container_width=True)
        
        # Create a bar chart of class-wise F1 scores
        class_rows = report_df.index[:-2] if 'Macro Avg' in report_df.index else report_df.index
        class_f1 = report_df.loc[class_rows, 'F1-Score']
        
        fig = px.bar(
            x=class_f1.index,
            y=class_f1.values,
            labels={'x': 'Class', 'y': 'F1-Score'},
            title='F1-Score by Class',
            color=class_f1.values,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            xaxis_title='Class',
            yaxis_title='F1-Score',
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation guidance
        with st.expander("Interpreting the Classification Report"):
            st.markdown("""
            ### Classification Report Components
            
            The classification report provides detailed metrics for each class:
            
            - **Precision**: Ability of the classifier not to label as positive a sample that is negative (TP / (TP + FP))
            - **Recall**: Ability of the classifier to find all the positive samples (TP / (TP + FN))
            - **F1-Score**: Harmonic mean of precision and recall (2 * (precision * recall) / (precision + recall))
            - **Support**: Number of actual occurrences of the class in the specified dataset
            
            ### Summary Rows
            
            - **Macro Avg**: Simple average of metrics for each class (treats all classes equally)
            - **Weighted Avg**: Average weighted by the number of samples in each class (accounts for class imbalance)
            
            ### Analysis Tips
            
            - Look for classes with lower F1-scores to identify where the model struggles
            - Compare macro and weighted averages to understand the impact of class imbalance
            - Classes with low support may have less reliable metrics
            """)
    
    # Export results
    st.subheader("Export Results")
    
    # Create a dictionary with all metrics
    metrics_dict = {
        "Basic Metrics": {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        },
        "Confusion Matrix": cm.tolist(),
        "Class Labels": class_labels,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Convert to JSON and offer download
    metrics_json = pd.Series(metrics_dict).to_json()
    
    # Create download link
    b64 = base64.b64encode(metrics_json.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="classification_metrics.json">Download Metrics (JSON)</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    # Offer report in CSV format as well
    csv_data = report_df.to_csv()
    b64_csv = base64.b64encode(csv_data.encode()).decode()
    href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="classification_report.csv">Download Classification Report (CSV)</a>'
    st.markdown(href_csv, unsafe_allow_html=True)

    with tab2:
        st.header("Regression Metrics Calculator")
        
        st.markdown("""
        Evaluate regression model performance with standard metrics and visualizations.
        You can either:
        1. Upload predictions from any model
        2. Choose from your previously trained models
        """)
        
        # Data source options
        data_source = st.radio(
            "Select data source:",
            options=["Upload predictions", "Use trained model"],
            horizontal=True,
            key="regression_data_source"
        )
        
        if data_source == "Upload predictions":
            st.subheader("Upload Predictions")
            
            # File uploader for predictions
            uploaded_file = st.file_uploader(
                "Upload CSV file with actual and predicted values", 
                type=["csv"], 
                key="regression_upload"
            )
            
            if uploaded_file is not None:
                try:
                    # Read the data
                    pred_df = pd.read_csv(uploaded_file)
                    
                    # Display preview
                    st.subheader("Data Preview")
                    st.dataframe(pred_df.head(), use_container_width=True)
                    
                    # Check if required columns exist
                    st.subheader("Column Selection")
                    st.markdown("Please select the columns containing actual and predicted values.")
                    
                    # Get column names
                    columns = pred_df.columns.tolist()
                    
                    # Select actual and predicted columns
                    actual_col = st.selectbox("Select column with actual values:", options=columns, key="reg_actual")
                    pred_col = st.selectbox("Select column with predicted values:", options=columns, index=min(1, len(columns)-1), key="reg_pred")
                    
                    # Check if the selected columns are valid
                    if actual_col and pred_col:
                        # Extract actual and predicted values
                        y_true = pred_df[actual_col].astype(float)
                        y_pred = pred_df[pred_col].astype(float)
                        
                        # Basic validation
                        if len(y_true) != len(y_pred):
                            st.error("Error: Actual and predicted values must have the same length.")
                        else:
                            # Continue with evaluation
                            st.subheader("Evaluate Model")
                            
                            if st.button("Calculate Metrics", key="reg_calc_button"):
                                # Calculate and display metrics
                                display_regression_metrics(y_true, y_pred, actual_col)
                
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    st.info("Please ensure your CSV file is properly formatted with numeric columns for actual and predicted values.")
        
        else:  # Use trained model
            st.subheader("Use Trained Model")
            
            # Check for trained models in session state
            models = []
            if 'lr_trained_model' in st.session_state and st.session_state.lr_trained_model is not None:
                models.append("Linear Regression")
            
            if not models:
                st.warning("No trained regression models found. Please train a model first.")
            else:
                # Select model
                selected_model = st.selectbox("Select trained model:", options=models, key="reg_model_select")
                
                # Get model data based on selection
                if selected_model == "Linear Regression" and 'lr_trained_model' in st.session_state:
                    y_true = st.session_state.lr_y_test
                    y_pred = st.session_state.lr_pipeline.predict(st.session_state.lr_X_test)
                    target_col = st.session_state.lr_target if 'lr_target' in st.session_state else "Target"
                
                # Calculate and display metrics
                if 'y_true' in locals() and 'y_pred' in locals():
                    st.subheader(f"Evaluation for {selected_model}")
                    
                    # Display information about the data
                    st.info(f"Using test data with {len(y_true)} samples.")
                    
                    # Calculate and display metrics
                    display_regression_metrics(y_true, y_pred, target_col)
                else:
                    st.error("Could not retrieve model data. Please retrain the model.")


def display_regression_metrics(y_true, y_pred, target_name="Target"):
    """
    Calculate and display regression metrics with visualizations.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth target values
    y_pred : array-like
        Estimated target values
    target_name : str
        Name of the target variable for labeling
    """
    # Create tabs for different metric groups
    metric_tab1, metric_tab2, metric_tab3 = st.tabs([
        "Error Metrics", 
        "Goodness of Fit", 
        "Residual Analysis"
    ])
    
    with metric_tab1:
        st.subheader("Error Metrics")
        
        # Calculate error metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Try to calculate MAPE (can fail if y_true contains zeros)
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # Convert to percentage
        except:
            # Calculate manually avoiding division by zero
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10))) * 100
        
        # Calculate median absolute error (more robust to outliers)
        median_ae = np.median(np.abs(y_true - y_pred))
        
        # Calculate mean absolute scaled error (MASE) if more than 10 samples
        if len(y_true) > 10:
            # Naively assume the data is ordered for simplicity
            # In a real time series, you would use the actual time ordering
            naive_forecast_errors = np.abs(y_true[1:] - y_true[:-1])
            naive_mae = np.mean(naive_forecast_errors)
            if naive_mae > 0:
                mase = mae / naive_mae
            else:
                mase = None
        else:
            mase = None
        
        # Display metrics in a grid
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
            st.caption("Average of squared differences between predicted and actual values")
            
            st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
            st.caption("Square root of MSE, interpretable in the same units as the target")
            
            st.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
            st.caption("Average of absolute differences between predicted and actual values")
        
        with col2:
            st.metric("Mean Absolute Percentage Error (MAPE)", f"{mape:.2f}%")
            st.caption("Average percentage difference between predicted and actual values")
            
            st.metric("Median Absolute Error", f"{median_ae:.4f}")
            st.caption("Median of absolute differences (robust to outliers)")
            
            if mase is not None:
                st.metric("Mean Absolute Scaled Error (MASE)", f"{mase:.4f}")
                st.caption("MAE relative to a naive forecast (values < 1 indicate model is better than naive)")
        
        # Interpretation guidance
        with st.expander("Interpreting Error Metrics"):
            st.markdown("""
            ### Error Metric Guide
            
            - **MSE (Mean Squared Error)**: Penalizes larger errors more heavily. Useful when large errors are particularly undesirable.
            
            - **RMSE (Root Mean Squared Error)**: Same as MSE but in the original units of the target variable. The most common error metric for regression.
            
            - **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual values. More robust to outliers than MSE/RMSE.
            
            - **MAPE (Mean Absolute Percentage Error)**: Expresses error as a percentage, making it scale-independent. Use with caution when actual values are close to zero.
            
            - **Median Absolute Error**: Similar to MAE but uses median instead of mean, making it even more robust to outliers.
            
            - **MASE (Mean Absolute Scaled Error)**: Compares the model against a naive baseline. Values less than 1 indicate the model is better than the naive approach.
            
            ### Which metric to use?
            
            - Use **RMSE** when large errors are more significant and the target variable has no outliers
            - Use **MAE** when you want to treat all error magnitudes equally
            - Use **MAPE** when comparing across different scales
            - Use **Median Absolute Error** when your data has outliers
            - Use **MASE** when evaluating time series forecasts
            """)
    
    with metric_tab2:
        st.subheader("Goodness of Fit Metrics")
        
        # Calculate goodness of fit metrics
        r2 = r2_score(y_true, y_pred)
        adjusted_r2 = 1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - 2 - 1)  # Assumes 2 features for simplicity
        explained_var = explained_variance_score(y_true, y_pred)
        
        # Display metrics in a grid
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("R² Score (Coefficient of Determination)", f"{r2:.4f}")
            st.caption("Proportion of variance explained by the model (0-1, higher is better)")
            
            st.metric("Adjusted R²", f"{adjusted_r2:.4f}")
            st.caption("R² adjusted for the number of predictors")
        
        with col2:
            st.metric("Explained Variance", f"{explained_var:.4f}")
            st.caption("Proportion of variance explained (similar to R²)")
            
            # Calculate and display R² quality assessment
            if r2 < 0:
                quality = "Poor (worse than mean predictor)"
                color = "red"
            elif r2 < 0.2:
                quality = "Very weak"
                color = "red"
            elif r2 < 0.4:
                quality = "Weak"
                color = "orange"
            elif r2 < 0.6:
                quality = "Moderate"
                color = "blue"
            elif r2 < 0.8:
                quality = "Strong"
                color = "green"
            else:
                quality = "Very strong"
                color = "green"
            
            st.markdown(f"**Model Fit Quality:** <span style='color:{color}'>{quality}</span>", unsafe_allow_html=True)
            st.caption("Qualitative assessment of the R² score")
        
        # Actual vs. Predicted Plot
        st.subheader("Actual vs. Predicted Values")
        
        pred_df = pd.DataFrame({
            'Actual': y_true,
            'Predicted': y_pred,
            'Error': y_true - y_pred
        })
        
        fig = px.scatter(
            pred_df,
            x='Actual',
            y='Predicted',
            color='Error',
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0,
            opacity=0.7,
            labels={'Actual': f'Actual {target_name}', 'Predicted': f'Predicted {target_name}'}
        )
        
        # Add perfect prediction line
        min_val = min(pred_df['Actual'].min(), pred_df['Predicted'].min())
        max_val = max(pred_df['Actual'].max(), pred_df['Predicted'].max())
        padding = (max_val - min_val) * 0.1
        
        fig.add_trace(
            go.Scatter(
                x=[min_val - padding, max_val + padding],
                y=[min_val - padding, max_val + padding],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='black', dash='dash')
            )
        )
        
        # Update layout for better visualization
        fig.update_layout(
            title='Actual vs. Predicted Values',
            xaxis_title=f'Actual {target_name}',
            yaxis_title=f'Predicted {target_name}',
            legend_title='Error',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation guidance
        with st.expander("Interpreting Goodness of Fit Metrics"):
            st.markdown("""
            ### Goodness of Fit Metrics Guide
            
            - **R² (Coefficient of Determination)**: Measures the proportion of variance in the dependent variable that is predictable from the independent variables.
              - R² = 1: Perfect fit, all variation explained
              - R² = 0: Model doesn't explain any variation (equivalent to predicting the mean)
              - R² < 0: Model performs worse than predicting the mean
            
            - **Adjusted R²**: Modifies R² to account for the number of predictors. Helps prevent overfitting by penalizing excessive features.
            
            - **Explained Variance**: Similar to R², but doesn't penalize for systematic bias in predictions.
            
            ### R² Quality Guide
            
            | R² Value | Quality Assessment |
            |----------|-------------------|
            | < 0      | Poor (worse than baseline) |
            | 0.0-0.2  | Very weak relationship |
            | 0.2-0.4  | Weak relationship |
            | 0.4-0.6  | Moderate relationship |
            | 0.6-0.8  | Strong relationship |
            | 0.8-1.0  | Very strong relationship |
            
            ### Actual vs. Predicted Plot
            
            - **Perfect fit**: Points would lie exactly on the dashed line
            - **Systematic bias**: Points consistently above or below the line
            - **Heteroscedasticity**: Fan-shaped pattern (error variance increases/decreases with prediction value)
            - **Outliers**: Points far from the line may represent unusual cases
            """)
    
    with metric_tab3:
        st.subheader("Residual Analysis")
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Residuals vs. Predicted plot
        st.subheader("Residuals vs. Predicted Values")
        
        residual_df = pd.DataFrame({
            'Predicted': y_pred,
            'Residuals': residuals,
            'Standardized Residuals': residuals / np.std(residuals)
        })
        
        fig_res = px.scatter(
            residual_df,
            x='Predicted',
            y='Residuals',
            opacity=0.7,
            labels={'Predicted': f'Predicted {target_name}', 'Residuals': 'Residuals'}
        )
        
        # Add horizontal line at y=0
        fig_res.add_hline(
            y=0,
            line_dash="dash",
            line_color="red",
            annotation_text="Zero Line",
            annotation_position="bottom right"
        )
        
        # Update layout
        fig_res.update_layout(
            title='Residuals vs. Predicted Values',
            xaxis_title=f'Predicted {target_name}',
            yaxis_title='Residuals',
            height=500
        )
        
        st.plotly_chart(fig_res, use_container_width=True)
        
        # Residual Distribution
        st.subheader("Residual Distribution")
        
        # Create histogram of residuals
        fig_hist = px.histogram(
            residual_df, 
            x='Residuals',
            marginal='box',
            histnorm='probability density',
            labels={'Residuals': 'Residuals'}
        )
        
        # Add normal distribution curve for comparison
        mean_residuals = np.mean(residuals)
        std_residuals = np.std(residuals)
        x_normal = np.linspace(mean_residuals - 4*std_residuals, mean_residuals + 4*std_residuals, 100)
        y_normal = 1/(std_residuals * np.sqrt(2 * np.pi)) * np.exp( - (x_normal - mean_residuals)**2 / (2 * std_residuals**2))
        
        fig_hist.add_trace(
            go.Scatter(
                x=x_normal,
                y=y_normal,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='red')
            )
        )
        
        # Update layout
        fig_hist.update_layout(
            title='Distribution of Residuals',
            xaxis_title='Residuals',
            yaxis_title='Density',
            height=500
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Q-Q Plot
        st.subheader("Normal Q-Q Plot")
        
        from scipy import stats
        
        # Prepare Q-Q plot data
        standardized_resid = residual_df['Standardized Residuals'].sort_values()
        n = len(standardized_resid)
        theoretical_quantiles = np.array([stats.norm.ppf((i + 0.5) / n) for i in range(n)])
        
        # Create Q-Q plot
        fig_qq = px.scatter(
            x=theoretical_quantiles,
            y=standardized_resid,
            labels={'x': 'Theoretical Quantiles', 'y': 'Standardized Residuals'}
        )
        
        # Add reference line
        min_q = min(theoretical_quantiles)
        max_q = max(theoretical_quantiles)
        fig_qq.add_trace(
            go.Scatter(
                x=[min_q, max_q],
                y=[min_q, max_q],
                mode='lines',
                name='Reference Line',
                line=dict(color='red', dash='dash')
            )
        )
        
        # Update layout
        fig_qq.update_layout(
            title='Normal Q-Q Plot',
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Standardized Residuals',
            height=500
        )
        
        st.plotly_chart(fig_qq, use_container_width=True)
        
        # Residual Statistics
        st.subheader("Residual Statistics")
        
        # Calculate Shapiro-Wilk test for normality
        shapiro_test = stats.shapiro(residuals)
        
        residual_stats = pd.DataFrame({
            'Statistic': [
                'Mean',
                'Standard Deviation',
                'Minimum',
                'Maximum',
                'Skewness',
                'Kurtosis',
                'Shapiro-Wilk p-value'
            ],
            'Value': [
                f"{np.mean(residuals):.6f}",
                f"{np.std(residuals):.4f}",
                f"{np.min(residuals):.4f}",
                f"{np.max(residuals):.4f}",
                f"{stats.skew(residuals):.4f}",
                f"{stats.kurtosis(residuals):.4f}",
                f"{shapiro_test[1]:.6f}"
            ]
        })
        
        st.dataframe(residual_stats, use_container_width=True)
        
        # Interpretation based on tests
        if abs(np.mean(residuals)) > 0.1 * np.std(residuals):
            st.warning("""
            **Non-zero Mean Residuals**: The mean of residuals is not close to zero. 
            This suggests a systematic bias in the predictions.
            """)
        else:
            st.success("**Mean Residuals**: Close to zero, which is good.")
        
        # Check for normality of residuals
        if shapiro_test[1] < 0.05:
            st.warning("""
            **Non-normal Residuals**: The Shapiro-Wilk test indicates that residuals are not normally distributed 
            (p < 0.05). This may affect the validity of confidence intervals and hypothesis tests.
            """)
        else:
            st.success("""
            **Normal Residuals**: The Shapiro-Wilk test suggests the residuals follow a normal distribution 
            (p ≥ 0.05), which is good for the validity of the regression model.
            """)
        
        # Check for heteroscedasticity (simple correlation check)
        corr_resid_pred = np.corrcoef(y_pred, np.abs(residuals))[0, 1]
        if abs(corr_resid_pred) > 0.2:
            st.warning(f"""
            **Potential Heteroscedasticity**: There appears to be a relationship between predicted values and 
            residual magnitudes (correlation = {corr_resid_pred:.4f}). This suggests non-constant variance 
            in the residuals, which violates a key assumption of linear regression.
            """)
        
        # Interpretation guidance
        with st.expander("Interpreting Residual Analysis"):
            st.markdown("""
            ### Residual Analysis Guide
            
            Residual analysis helps assess whether the regression model meets key assumptions:
            
            ### 1. Residuals vs. Predicted Plot
            
            Ideal characteristics:
            - **Random scatter around zero line**: No patterns or trends
            - **Equal spread across range**: Constant variance (homoscedasticity)
            
            Common issues:
            - **Funnel shape**: Heteroscedasticity (non-constant variance)
            - **Curved pattern**: Non-linear relationship missed by model
            - **Trend in residuals**: Systematic bias in predictions
            
            ### 2. Residual Distribution
            
            Ideal characteristics:
            - **Bell-shaped and centered at zero**: Normal distribution
            - **Symmetrical**: Equal spread on both sides of zero
            
            Common issues:
            - **Skewed distribution**: Potential model misspecification
            - **Heavy tails**: Outliers present in the data
            - **Multiple peaks**: Possible multimodal data or mixed populations
            
            ### 3. Q-Q Plot
            
            Ideal characteristic:
            - **Points follow the diagonal line**: Residuals are normally distributed
            
            Common issues:
            - **S-shaped curve**: Skewed residuals
            - **Points off the line at ends**: Heavy tails (outliers)
            
            ### 4. Statistical Tests
            
            - **Shapiro-Wilk test**: Tests for normality of residuals (p ≥ 0.05 suggests normality)
            - **Mean of residuals**: Should be close to zero
            - **Skewness**: Measure of asymmetry (should be close to 0)
            - **Kurtosis**: Measure of "tailedness" (should be close to 0 for normal distribution)
            """)
    
    # Export results
    st.subheader("Export Results")
    
    # Create a dictionary with all metrics
    metrics_dict = {
        "Error Metrics": {
            "MSE": float(mse),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "MAPE": float(mape)
        },
        "Goodness of Fit": {
            "R-squared": float(r2),
            "Adjusted R-squared": float(adjusted_r2),
            "Explained Variance": float(explained_var)
        },
        "Residual Statistics": {
            "Mean": float(np.mean(residuals)),
            "Standard Deviation": float(np.std(residuals)),
            "Shapiro-Wilk p-value": float(shapiro_test[1])
        },
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Convert to JSON and offer download
    metrics_json = pd.Series(metrics_dict).to_json()
    
    # Create download link
    b64 = base64.b64encode(metrics_json.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="regression_metrics.json">Download Metrics (JSON)</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    # Offer CSV download of predictions and residuals
    csv_data = pred_df.to_csv(index=False)
    b64_csv = base64.b64encode(csv_data.encode()).decode()
    href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="regression_predictions.csv">Download Predictions with Residuals (CSV)</a>'
    st.markdown(href_csv, unsafe_allow_html=True)

# This is outside any tab context - check if data is loaded
if df is None:
    show_file_required_warning()
else:
    # Only display tab3 content if data is loaded
    with tab3:
        st.header("Custom Evaluation")
        
        st.markdown("""
        Define your own custom metrics and evaluation criteria.
        This section allows you to create specialized visualizations and metrics 
        for your specific use case.
        """)
        
        st.subheader("Data Source")
        
        # Data input options
        data_input = st.radio(
            "Select data source:",
            options=["Upload custom data", "Generate sample data"],
            horizontal=True
        )
        
        if data_input == "Upload custom data":
            # File uploader
            uploaded_file = st.file_uploader(
                "Upload CSV file with your data", 
                type=["csv"], 
                key="custom_eval_upload"
            )
            
            if uploaded_file is not None:
                try:
                    # Read the data
                    custom_df = pd.read_csv(uploaded_file)
                    
                    # Display preview
                    st.subheader("Data Preview")
                    st.dataframe(custom_df.head(), use_container_width=True)
                    
                    # Continue with custom evaluation
                    if not custom_df.empty:
                        st.session_state.custom_eval_df = custom_df
                        st.success("Data loaded successfully!")
                
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        else:  # Generate sample data
            st.subheader("Generate Sample Data")
            
            # Data generation options
            data_type = st.selectbox(
                "Select the type of data to generate:",
                options=["Regression", "Classification", "Time Series"]
            )
            
            # Number of samples
            n_samples = st.slider("Number of samples:", min_value=50, max_value=1000, value=200, step=50)
            
            # Noise level
            noise = st.slider("Noise level:", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
            
            # Generate button
            if st.button("Generate Data"):
                if data_type == "Regression":
                    # Generate regression data
                    np.random.seed(42)
                    X = np.random.rand(n_samples, 2)
                    y_true = 3*X[:, 0] + 2*X[:, 1] + np.random.normal(0, noise, size=n_samples)
                    y_pred = 3*X[:, 0] + 2*X[:, 1] + np.random.normal(0, noise*1.2, size=n_samples)
                    
                    custom_df = pd.DataFrame({
                        'feature1': X[:, 0],
                        'feature2': X[:, 1],
                        'actual': y_true,
                        'predicted': y_pred,
                        'residual': y_true - y_pred
                    })
                
                elif data_type == "Classification":
                    # Generate classification data
                    np.random.seed(42)
                    X = np.random.rand(n_samples, 2)
                    y_true = (0.5*X[:, 0] + 0.5*X[:, 1] > 0.5).astype(int)
                    
                    # Add some noise to predictions
                    prob_correct = 0.8 - noise * 0.3  # Higher noise = lower probability of correct prediction
                    y_pred = np.copy(y_true)
                    flip_idx = np.random.rand(n_samples) > prob_correct
                    y_pred[flip_idx] = 1 - y_pred[flip_idx]
                    
                    # Probabilities
                    probs = np.random.rand(n_samples)
                    probs[y_pred == 1] = 0.5 + probs[y_pred == 1] * 0.5
                    probs[y_pred == 0] = probs[y_pred == 0] * 0.5
                    
                    custom_df = pd.DataFrame({
                        'feature1': X[:, 0],
                        'feature2': X[:, 1],
                        'actual_class': y_true,
                        'predicted_class': y_pred,
                        'probability': probs
                    })
                
                else:  # Time Series
                    # Generate time series data
                    np.random.seed(42)
                    dates = pd.date_range(start='2023-01-01', periods=n_samples)
                    
                    # Create trend, seasonality, and noise
                    trend = np.linspace(0, 3, n_samples)
                    seasonality = 2 * np.sin(np.linspace(0, 4*np.pi, n_samples))
                    noise_component = np.random.normal(0, noise, size=n_samples)
                    
                    # Combine components
                    y_true = trend + seasonality + noise_component
                    
                    # Create predictions with slightly different parameters
                    pred_noise = np.random.normal(0, noise, size=n_samples)
                    y_pred = trend + seasonality * 0.9 + pred_noise
                    
                    custom_df = pd.DataFrame({
                        'date': dates,
                        'actual': y_true,
                        'predicted': y_pred,
                        'residual': y_true - y_pred
                    })
                
                st.session_state.custom_eval_df = custom_df
                st.success("Sample data generated successfully!")
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(custom_df.head(), use_container_width=True)
        
        # Check if data is available for custom evaluation
        if 'custom_eval_df' in st.session_state and not st.session_state.custom_eval_df.empty:
            custom_df = st.session_state.custom_eval_df
            
            st.subheader("Custom Evaluation Tools")
            
            # Create tabs for different custom evaluation tools
            custom_tab1, custom_tab2, custom_tab3 = st.tabs([
                "Custom Visualization", 
                "Custom Metrics", 
                "Data Exploration"
            ])
            
            with custom_tab1:
                st.subheader("Custom Visualization")
                
                # Get columns for visualization
                numeric_cols = custom_df.select_dtypes(include=[np.number]).columns.tolist()
                date_cols = [col for col in custom_df.columns if custom_df[col].dtype == 'datetime64[ns]']
                all_cols = custom_df.columns.tolist()
                
                # Plot type selection
                plot_type = st.selectbox(
                    "Select visualization type:",
                    options=["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Heatmap"]
                )
                
                if plot_type == "Scatter Plot":
                    # Column selection for scatter plot
                    x_col = st.selectbox("X-axis:", options=all_cols, key="scatter_x")
                    y_col = st.selectbox("Y-axis:", options=numeric_cols, key="scatter_y")
                    
                    # Optional color column
                    color_option = st.checkbox("Add color dimension")
                    color_col = None
                    if color_option:
                        color_col = st.selectbox("Color by:", options=all_cols)
                    
                    # Create plot
                    if st.button("Generate Scatter Plot"):
                        fig = px.scatter(
                            custom_df, 
                            x=x_col, 
                            y=y_col,
                            color=color_col,
                            title=f"{y_col} vs {x_col}",
                            labels={x_col: x_col, y_col: y_col},
                            opacity=0.7
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                elif plot_type == "Line Chart":
                    # Column selection for line chart
                    x_col = st.selectbox("X-axis:", options=all_cols, key="line_x")
                    
                    # Allow multiple Y columns
                    y_cols = st.multiselect("Y-axis (multiple):", options=numeric_cols, key="line_y")
                    
                    # Create plot
                    if st.button("Generate Line Chart") and y_cols:
                        fig = px.line(
                            custom_df, 
                            x=x_col, 
                            y=y_cols,
                            title=f"Line Chart",
                            labels={y_col: y_col for y_col in y_cols}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                elif plot_type == "Bar Chart":
                    # Column selection for bar chart
                    x_col = st.selectbox("X-axis (categories):", options=all_cols, key="bar_x")
                    y_col = st.selectbox("Y-axis (values):", options=numeric_cols, key="bar_y")
                    
                    # Optional color column
                    color_option = st.checkbox("Add color dimension", key="bar_color_opt")
                    color_col = None
                    if color_option:
                        color_col = st.selectbox("Color by:", options=all_cols, key="bar_color")
                    
                    # Create plot
                    if st.button("Generate Bar Chart"):
                        fig = px.bar(
                            custom_df, 
                            x=x_col, 
                            y=y_col,
                            color=color_col,
                            title=f"Bar Chart of {y_col} by {x_col}",
                            labels={x_col: x_col, y_col: y_col}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                elif plot_type == "Histogram":
                    # Column selection for histogram
                    col = st.selectbox("Select column:", options=numeric_cols, key="hist_col")
                    
                    # Histogram options
                    bins = st.slider("Number of bins:", min_value=5, max_value=100, value=20)
                    
                    # Create plot
                    if st.button("Generate Histogram"):
                        fig = px.histogram(
                            custom_df, 
                            x=col,
                            nbins=bins,
                            marginal="box",
                            title=f"Histogram of {col}",
                            labels={col: col}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                elif plot_type == "Box Plot":
                    # Column selection for box plot
                    y_col = st.selectbox("Values:", options=numeric_cols, key="box_y")
                    
                    # Optional category column
                    category_option = st.checkbox("Group by category")
                    x_col = None
                    if category_option:
                        x_col = st.selectbox("Category:", options=all_cols, key="box_x")
                    
                    # Create plot
                    if st.button("Generate Box Plot"):
                        fig = px.box(
                            custom_df, 
                            x=x_col, 
                            y=y_col,
                            title=f"Box Plot of {y_col}" + (f" by {x_col}" if x_col else ""),
                            labels={x_col: x_col, y_col: y_col} if x_col else {y_col: y_col}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                elif plot_type == "Heatmap":
                    # Use only numeric columns for correlation
                    if len(numeric_cols) < 2:
                        st.warning("Need at least 2 numeric columns for a heatmap.")
                    else:
                        # Correlation method
                        corr_method = st.selectbox(
                            "Correlation method:",
                            options=["pearson", "spearman", "kendall"],
                            key="heatmap_method"
                        )
                        
                        # Create plot
                        if st.button("Generate Heatmap"):
                            # Calculate correlation matrix
                            corr_matrix = custom_df[numeric_cols].corr(method=corr_method)
                            
                            # Create heatmap
                            fig = px.imshow(
                                corr_matrix,
                                text_auto='.2f',
                                color_continuous_scale='RdBu_r',
                                title=f"{corr_method.capitalize()} Correlation Heatmap"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
            
            with custom_tab2:
                st.subheader("Custom Metrics")
                
                # Get columns for metrics
                numeric_cols = custom_df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_cols) < 2:
                    st.warning("Need at least 2 numeric columns to calculate metrics.")
                else:
                    # Column selection for actual and predicted values
                    actual_col = st.selectbox("Actual values:", options=numeric_cols, key="metric_actual")
                    pred_col = st.selectbox("Predicted values:", options=numeric_cols, key="metric_pred")
                    
                    # Problem type
                    problem_type = st.radio(
                        "Problem type:",
                        options=["Regression", "Binary Classification"],
                        horizontal=True
                    )
                    
                    # Custom metrics options
                    st.subheader("Select Metrics to Calculate")
                    
                    if problem_type == "Regression":
                        # Regression metrics options
                        metrics_to_calc = {}
                        metrics_to_calc["MSE"] = st.checkbox("Mean Squared Error", value=True)
                        metrics_to_calc["RMSE"] = st.checkbox("Root Mean Squared Error", value=True)
                        metrics_to_calc["MAE"] = st.checkbox("Mean Absolute Error", value=True)
                        metrics_to_calc["MAPE"] = st.checkbox("Mean Absolute Percentage Error", value=True)
                        metrics_to_calc["R2"] = st.checkbox("R² Score", value=True)
                        metrics_to_calc["ExplainedVar"] = st.checkbox("Explained Variance", value=True)
                        metrics_to_calc["MaxError"] = st.checkbox("Maximum Error", value=False)
                        metrics_to_calc["MedianAE"] = st.checkbox("Median Absolute Error", value=False)
                        
                        # Custom threshold metrics
                        threshold_option = st.checkbox("Add threshold-based metrics")
                        threshold = None
                        if threshold_option:
                            threshold = st.number_input("Error threshold:", value=1.0)
                            metrics_to_calc["ThresholdError"] = True
                        
                        # Calculate metrics button
                        if st.button("Calculate Selected Metrics"):
                            # Get values
                            y_true = custom_df[actual_col].values
                            y_pred = custom_df[pred_col].values
                            
                            # Calculate selected metrics
                            results = {}
                            
                            if metrics_to_calc.get("MSE", False):
                                results["Mean Squared Error"] = mean_squared_error(y_true, y_pred)
                            
                            if metrics_to_calc.get("RMSE", False):
                                results["Root Mean Squared Error"] = np.sqrt(mean_squared_error(y_true, y_pred))
                            
                            if metrics_to_calc.get("MAE", False):
                                results["Mean Absolute Error"] = mean_absolute_error(y_true, y_pred)
                            
                            if metrics_to_calc.get("MAPE", False):
                                # Handle zeros in y_true
                                valid_indices = y_true != 0
                                if np.any(valid_indices):
                                    results["Mean Absolute Percentage Error"] = np.mean(
                                        np.abs((y_true[valid_indices] - y_pred[valid_indices]) / y_true[valid_indices])
                                    ) * 100
                                else:
                                    results["Mean Absolute Percentage Error"] = "N/A (division by zero)"
                            
                            if metrics_to_calc.get("R2", False):
                                results["R² Score"] = r2_score(y_true, y_pred)
                            
                            if metrics_to_calc.get("ExplainedVar", False):
                                results["Explained Variance"] = explained_variance_score(y_true, y_pred)
                            
                            if metrics_to_calc.get("MaxError", False):
                                results["Maximum Error"] = np.max(np.abs(y_true - y_pred))
                            
                            if metrics_to_calc.get("MedianAE", False):
                                results["Median Absolute Error"] = np.median(np.abs(y_true - y_pred))
                            
                            if metrics_to_calc.get("ThresholdError", False) and threshold is not None:
                                # Calculate percentage of predictions within threshold
                                within_threshold = np.abs(y_true - y_pred) <= threshold
                                results[f"% Within ±{threshold}"] = np.mean(within_threshold) * 100
                            
                            # Display results
                            results_df = pd.DataFrame({
                                'Metric': list(results.keys()),
                                'Value': [f"{v:.4f}" if isinstance(v, (int, float)) else v for v in results.values()]
                            })
                            
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Visualization of errors
                            st.subheader("Error Distribution")
                            
                            errors = y_true - y_pred
                            
                            fig = px.histogram(
                                errors,
                                title="Distribution of Errors",
                                labels={'value': 'Error'},
                                marginal="box"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    else:  # Binary Classification
                        # Check if data seems appropriate for binary classification
                        y_true = custom_df[actual_col].values
                        y_pred = custom_df[pred_col].values
                        
                        unique_true = np.unique(y_true)
                        unique_pred = np.unique(y_pred)
                        
                        if len(unique_true) > 2 or len(unique_pred) > 2:
                            st.warning(
                                "The selected columns contain more than 2 unique values. " +
                                "For binary classification, data should contain only 2 classes (typically 0 and 1)."
                            )
                        
                        # Binary classification metrics options
                        metrics_to_calc = {}
                        metrics_to_calc["Accuracy"] = st.checkbox("Accuracy", value=True)
                        metrics_to_calc["Precision"] = st.checkbox("Precision", value=True)
                        metrics_to_calc["Recall"] = st.checkbox("Recall", value=True)
                        metrics_to_calc["F1"] = st.checkbox("F1 Score", value=True)
                        metrics_to_calc["ConfMatrix"] = st.checkbox("Confusion Matrix", value=True)
                        
                        # Threshold for binary predictions
                        threshold_option = st.checkbox("Apply custom threshold to predictions")
                        threshold = 0.5
                        if threshold_option:
                            threshold = st.slider("Classification threshold:", min_value=0.0, max_value=1.0, value=0.5)
                            st.info(f"Values >= {threshold} will be classified as positive (1)")
                        
                        # Calculate metrics button
                        if st.button("Calculate Selected Metrics"):
                            try:
                                # Convert to binary if needed
                                if threshold_option:
                                    y_pred_binary = (y_pred >= threshold).astype(int)
                                else:
                                    y_pred_binary = y_pred
                                
                                # Ensure binary values
                                y_true_binary = y_true.astype(int)
                                y_pred_binary = y_pred_binary.astype(int)
                                
                                # Calculate selected metrics
                                results = {}
                                
                                if metrics_to_calc.get("Accuracy", False):
                                    results["Accuracy"] = accuracy_score(y_true_binary, y_pred_binary)
                                
                                if metrics_to_calc.get("Precision", False):
                                    results["Precision"] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                                
                                if metrics_to_calc.get("Recall", False):
                                    results["Recall"] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                                
                                if metrics_to_calc.get("F1", False):
                                    results["F1 Score"] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                                
                                # Display results
                                results_df = pd.DataFrame({
                                    'Metric': list(results.keys()),
                                    'Value': [f"{v:.4f}" for v in results.values()]
                                })
                                
                                st.dataframe(results_df, use_container_width=True)
                                
                                # Display confusion matrix if selected
                                if metrics_to_calc.get("ConfMatrix", False):
                                    st.subheader("Confusion Matrix")
                                    
                                    cm = confusion_matrix(y_true_binary, y_pred_binary)
                                    
                                    fig = px.imshow(
                                        cm,
                                        text_auto=True,
                                        labels=dict(x="Predicted", y="Actual", color="Count"),
                                        x=['Negative (0)', 'Positive (1)'],
                                        y=['Negative (0)', 'Positive (1)']
                                    )
                                    
                                    fig.update_layout(
                                        title="Confusion Matrix",
                                        xaxis_title="Predicted Class",
                                        yaxis_title="Actual Class"
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            except Exception as e:
                                st.error(f"Error calculating metrics: {str(e)}")
                                st.info("Make sure your data is appropriate for binary classification (values should be 0/1 or convertible to binary).")
            
            with custom_tab3:
                st.subheader("Data Exploration")
                
                # Summary statistics
                st.subheader("Summary Statistics")
                
                stat_options = st.multiselect(
                    "Select statistics to display:",
                    options=["Count", "Mean", "Standard Deviation", "Min", "25%", "Median (50%)", "75%", "Max"],
                    default=["Count", "Mean", "Standard Deviation", "Min", "Max"]
                )
                
                # Map selected options to pandas describe() names
                stat_map = {
                    "Count": "count",
                    "Mean": "mean",
                    "Standard Deviation": "std",
                    "Min": "min",
                    "25%": "25%",
                    "Median (50%)": "50%",
                    "75%": "75%",
                    "Max": "max"
                }
                
                if stat_options:
                    # Get stats indices based on selection
                    selected_stats = [stat_map[opt] for opt in stat_options]
                    
                    # Calculate and display statistics
                    desc_df = custom_df.describe().loc[selected_stats]
                    st.dataframe(desc_df, use_container_width=True)
                
                # Correlation analysis
                if len(numeric_cols) >= 2:
                    st.subheader("Correlation Analysis")
                    
                    corr_method = st.selectbox(
                        "Correlation method:",
                        options=["pearson", "spearman", "kendall"]
                    )
                    
                    # Calculate correlations
                    corr_matrix = custom_df[numeric_cols].corr(method=corr_method)
                    
                    # Display correlation matrix
                    st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'), use_container_width=True)
                    
                    # Highlight highest correlations
                    st.subheader("Highest Correlations")
                    
                    # Get upper triangle of correlation matrix (to avoid duplicates)

# Footer
create_footer() 
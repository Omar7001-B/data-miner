import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import io
import base64
from datetime import datetime
import os
import json
import scipy.stats as stats
from sklearn.model_selection import learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error
)

from utils import (
    load_css, 
    create_footer, 
    show_file_required_warning, 
    display_dataset_info
)

# Page config
st.set_page_config(page_title="Model Visualization", page_icon="📊", layout="wide")
load_css()

# Main title
st.title("Model Visualization 📊")
st.markdown("""
This page provides advanced visualization tools for analyzing and comparing your machine learning models.
Use these tools to gain deeper insights into model performance, understand feature importance,
and compare different models to make better decisions.
""")

# Check if data is loaded
df = st.session_state.get('df', None)

if df is not None:
    # Display dataset info
    display_dataset_info()
    
    # Check for trained models
    models = {}
    
    # Check for classification models
    if 'lr_trained_model' in st.session_state and st.session_state.lr_trained_model is not None:
        if hasattr(st.session_state.lr_trained_model, 'predict_proba'):
            models["Logistic Regression"] = {
                "model": st.session_state.lr_trained_model,
                "pipeline": st.session_state.lr_pipeline,
                "features": st.session_state.lr_feature_names,
                "target": st.session_state.lr_target,
                "type": "classification",
                "classes": st.session_state.lr_classes if 'lr_classes' in st.session_state else None,
                "X_test": st.session_state.get('lr_X_test', None),
                "y_test": st.session_state.get('lr_y_test', None),
                "metrics": st.session_state.get('lr_metrics', {}),
                "training_time": st.session_state.get('lr_training_time', None)
            }
    
    if 'dt_trained_model' in st.session_state and st.session_state.dt_trained_model is not None:
        models["Decision Tree"] = {
            "model": st.session_state.dt_trained_model,
            "pipeline": st.session_state.dt_pipeline,
            "features": st.session_state.dt_feature_names,
            "target": st.session_state.dt_target,
            "type": "classification",
            "classes": st.session_state.dt_classes if 'dt_classes' in st.session_state else None,
            "X_test": st.session_state.get('dt_X_test', None),
            "y_test": st.session_state.get('dt_y_test', None),
            "metrics": st.session_state.get('dt_metrics', {}),
            "training_time": st.session_state.get('dt_training_time', None)
        }
    
    if 'knn_trained_model' in st.session_state and st.session_state.knn_trained_model is not None:
        models["K-Nearest Neighbors"] = {
            "model": st.session_state.knn_trained_model,
            "pipeline": st.session_state.knn_pipeline,
            "features": st.session_state.knn_feature_names,
            "target": st.session_state.knn_target,
            "type": "classification",
            "classes": st.session_state.knn_classes if 'knn_classes' in st.session_state else None,
            "X_test": st.session_state.get('knn_X_test', None),
            "y_test": st.session_state.get('knn_y_test', None),
            "metrics": st.session_state.get('knn_metrics', {}),
            "training_time": st.session_state.get('knn_training_time', None)
        }
    
    # Check for regression models
    if 'lr_trained_model' in st.session_state and st.session_state.lr_trained_model is not None:
        # Linear regression has same session state key as logistic regression
        # Check the model type to differentiate
        if not hasattr(st.session_state.lr_trained_model, 'predict_proba'):
            # This is likely linear regression
            models["Linear Regression"] = {
                "model": st.session_state.lr_trained_model,
                "pipeline": st.session_state.lr_pipeline,
                "features": st.session_state.lr_feature_names,
                "target": st.session_state.lr_target,
                "type": "regression",
                "classes": None,
                "X_test": st.session_state.get('lr_X_test', None),
                "y_test": st.session_state.get('lr_y_test', None),
                "metrics": st.session_state.get('lr_metrics', {}),
                "training_time": st.session_state.get('lr_training_time', None)
            }
    
    # Group models by type
    classification_models = {k: v for k, v in models.items() if v["type"] == "classification"}
    regression_models = {k: v for k, v in models.items() if v["type"] == "regression"}
    
    if not models:
        st.warning("No trained models found. Please train at least one model first.")
    else:
        # Create tabs for visualization tools
        tab1, tab2, tab3 = st.tabs(["Model Comparison", "Feature Importance", "Learning Curves"])
        
        with tab1:
            st.header("Model Comparison")
            
            st.markdown("""
            Compare performance metrics across different models to identify the best performer for your task.
            You can compare models of the same type (classification or regression).
            """)
            
            # Model type selection
            model_type = st.radio(
                "Select model type to compare",
                options=["Classification", "Regression"],
                horizontal=True,
                disabled=len(classification_models) == 0 or len(regression_models) == 0
            )
            
            if model_type == "Classification" and classification_models:
                # Select models to compare
                model_options = list(classification_models.keys())
                selected_models = st.multiselect(
                    "Select models to compare",
                    options=model_options,
                    default=model_options[:min(3, len(model_options))]
                )
                
                if selected_models:
                    # Collect metrics for selected models
                    comparison_data = []
                    
                    for model_name in selected_models:
                        model_info = classification_models[model_name]
                        metrics = model_info["metrics"]
                        
                        # Basic metrics
                        model_data = {
                            "Model": model_name,
                            "Accuracy": metrics.get("accuracy", 0),
                            "Precision": metrics.get("precision", 0),
                            "Recall": metrics.get("recall", 0),
                            "F1 Score": metrics.get("f1", 0),
                            "Training Time (s)": model_info["training_time"] if model_info["training_time"] else 0
                        }
                        
                        comparison_data.append(model_data)
                    
                    # Create comparison dataframe
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Display comparison table
                    st.subheader("Metrics Comparison")
                    st.dataframe(comparison_df.set_index("Model").style.highlight_max(axis=0), use_container_width=True)
                    
                    # Create radar chart for comparing models
                    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score"]
                    
                    fig = go.Figure()
                    
                    for model_name in selected_models:
                        model_metrics = comparison_df[comparison_df["Model"] == model_name][metrics_to_plot].values[0]
                        
                        fig.add_trace(go.Scatterpolar(
                            r=model_metrics,
                            theta=metrics_to_plot,
                            fill='toself',
                            name=model_name
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )),
                        showlegend=True,
                        title="Model Performance Comparison"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create bar charts for each metric
                    st.subheader("Individual Metrics Comparison")
                    
                    metric_tabs = st.tabs(metrics_to_plot + ["Training Time"])
                    
                    for i, metric in enumerate(metrics_to_plot):
                        with metric_tabs[i]:
                            fig = px.bar(
                                comparison_df,
                                x="Model",
                                y=metric,
                                title=f"{metric} Comparison",
                                color="Model",
                                text_auto='.3f'
                            )
                            
                            fig.update_layout(yaxis_range=[0, 1])
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Training time comparison
                    with metric_tabs[4]:
                        fig = px.bar(
                            comparison_df,
                            x="Model",
                            y="Training Time (s)",
                            title="Training Time Comparison",
                            color="Model",
                            text_auto='.3f'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            elif model_type == "Regression" and regression_models:
                # Select models to compare
                model_options = list(regression_models.keys())
                selected_models = st.multiselect(
                    "Select models to compare",
                    options=model_options,
                    default=model_options[:min(3, len(model_options))]
                )
                
                if selected_models:
                    # Collect metrics for selected models
                    comparison_data = []
                    
                    for model_name in selected_models:
                        model_info = regression_models[model_name]
                        metrics = model_info["metrics"]
                        
                        # Basic metrics
                        model_data = {
                            "Model": model_name,
                            "R² Score": metrics.get("r2", 0),
                            "MSE": metrics.get("mse", 0),
                            "RMSE": metrics.get("rmse", 0),
                            "MAE": metrics.get("mae", 0),
                            "Training Time (s)": model_info["training_time"] if model_info["training_time"] else 0
                        }
                        
                        comparison_data.append(model_data)
                    
                    # Create comparison dataframe
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Display comparison table
                    st.subheader("Metrics Comparison")
                    
                    # Different highlighting for different metrics (higher is better for R², lower is better for errors)
                    highlight_df = comparison_df.copy()
                    st.dataframe(
                        highlight_df.set_index("Model").style
                        .highlight_max(subset=["R² Score"], axis=0)
                        .highlight_min(subset=["MSE", "RMSE", "MAE"], axis=0),
                        use_container_width=True
                    )
                    
                    # Create bar charts for each metric
                    st.subheader("Individual Metrics Comparison")
                    
                    metric_tabs = st.tabs(["R² Score", "Error Metrics", "Training Time"])
                    
                    with metric_tabs[0]:
                        fig = px.bar(
                            comparison_df,
                            x="Model",
                            y="R² Score",
                            title="R² Score Comparison (higher is better)",
                            color="Model",
                            text_auto='.3f'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with metric_tabs[1]:
                        # Melt the dataframe to get a format suitable for a grouped bar chart
                        error_metrics = ["MSE", "RMSE", "MAE"]
                        melted_df = pd.melt(
                            comparison_df,
                            id_vars=["Model"],
                            value_vars=error_metrics,
                            var_name="Metric",
                            value_name="Value"
                        )
                        
                        fig = px.bar(
                            melted_df,
                            x="Model",
                            y="Value",
                            color="Metric",
                            barmode="group",
                            title="Error Metrics Comparison (lower is better)",
                            text_auto='.3f'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Training time comparison
                    with metric_tabs[2]:
                        fig = px.bar(
                            comparison_df,
                            x="Model",
                            y="Training Time (s)",
                            title="Training Time Comparison",
                            color="Model",
                            text_auto='.3f'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info(f"No {model_type.lower()} models found. Please train at least one {model_type.lower()} model first.")
        
        with tab2:
            st.header("Feature Importance")
            
            st.markdown("""
            Analyze feature importance across different models to understand which features 
            are most influential in making predictions.
            """)
            
            # Model selection for feature importance
            model_options = list(models.keys())
            selected_model = st.selectbox(
                "Select model to analyze",
                options=model_options
            )
            
            if selected_model:
                model_info = models[selected_model]
                
                # Extract feature importance if available
                model_type = model_info["type"]
                feature_names = model_info["features"]
                
                # Try to get feature importance
                feature_importance = None
                model = model_info["model"]
                
                try:
                    # For tree-based models
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = model.feature_importances_
                    
                    # For linear models
                    elif hasattr(model, 'coef_'):
                        if model_type == "classification" and len(model.coef_.shape) > 1:
                            # Multiclass case - take average of absolute coefficients
                            feature_importance = np.mean(np.abs(model.coef_), axis=0)
                        else:
                            # Binary classification or regression
                            feature_importance = np.abs(model.coef_)
                    
                    # For some pipeline cases where model is nested
                    elif hasattr(model, 'steps'):
                        for _, step in model.steps:
                            if hasattr(step, 'feature_importances_'):
                                feature_importance = step.feature_importances_
                                break
                            elif hasattr(step, 'coef_'):
                                if model_type == "classification" and len(step.coef_.shape) > 1:
                                    feature_importance = np.mean(np.abs(step.coef_), axis=0)
                                else:
                                    feature_importance = np.abs(step.coef_)
                                break
                
                except Exception as e:
                    st.error(f"Error extracting feature importance: {str(e)}")
                
                if feature_importance is not None and len(feature_importance) == len(feature_names):
                    # Create feature importance dataframe
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': feature_importance
                    })
                    
                    # Sort by importance
                    importance_df = importance_df.sort_values('Importance', ascending=False)
                    
                    # Display feature importance table
                    st.subheader("Feature Importance Ranking")
                    st.dataframe(importance_df, use_container_width=True)
                    
                    # Create bar chart
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title=f"Feature Importance for {selected_model}",
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature correlation with target
                    if model_info["target"] in df.columns:
                        st.subheader("Feature Correlation with Target")
                        
                        # Compute correlations with target
                        corr_data = df[feature_names + [model_info["target"]]].copy()
                        
                        # Only include numeric features
                        numeric_features = corr_data.select_dtypes(include=['number']).columns.tolist()
                        
                        if model_info["target"] in numeric_features and len(numeric_features) > 1:
                            # Compute correlations
                            correlations = corr_data[numeric_features].corr()[model_info["target"]].drop(model_info["target"])
                            
                            # Create dataframe for display
                            corr_df = pd.DataFrame({
                                'Feature': correlations.index,
                                'Correlation': correlations.values
                            }).sort_values('Correlation', key=abs, ascending=False)
                            
                            # Display correlation table
                            st.dataframe(corr_df, use_container_width=True)
                            
                            # Create bar chart
                            fig = px.bar(
                                corr_df,
                                x='Correlation',
                                y='Feature',
                                orientation='h',
                                title=f"Feature Correlation with {model_info['target']}",
                                color='Correlation',
                                color_continuous_scale='RdBu',
                                color_continuous_midpoint=0
                            )
                            
                            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Target variable or features are not numeric. Correlation analysis requires numeric data.")
                
                else:
                    st.info(f"Feature importance is not available for this model ({selected_model}).")
                    
                    # Provide alternative visualization for model types without direct feature importance
                    if model_type == "classification" and 'knn' in selected_model.lower():
                        st.subheader("Feature Analysis for KNN")
                        st.markdown("""
                        K-Nearest Neighbors doesn't provide direct feature importance scores.
                        However, we can analyze feature distributions to understand their impact.
                        """)
                        
                        # Feature selection for analysis
                        feature_for_analysis = st.selectbox(
                            "Select a feature to analyze",
                            options=feature_names
                        )
                        
                        if feature_for_analysis and model_info["target"] in df.columns:
                            # Analyze feature distribution by target class
                            if pd.api.types.is_numeric_dtype(df[feature_for_analysis]):
                                # For numeric features
                                fig = px.histogram(
                                    df,
                                    x=feature_for_analysis,
                                    color=model_info["target"],
                                    marginal="box",
                                    title=f"Distribution of {feature_for_analysis} by {model_info['target']}",
                                    barmode="overlay",
                                    opacity=0.7
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                # For categorical features
                                fig = px.histogram(
                                    df,
                                    x=feature_for_analysis,
                                    color=model_info["target"],
                                    title=f"Distribution of {feature_for_analysis} by {model_info['target']}",
                                    barmode="group"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Show feature pair correlation if it's a KNN model
                            st.subheader("Feature Pair Analysis")
                            st.markdown(
                                "In KNN, the relationship between pairs of features can be important for classification."
                            )
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                feature_x = st.selectbox(
                                    "Select X-axis feature",
                                    options=feature_names,
                                    key="feature_x"
                                )
                            
                            with col2:
                                remaining_features = [f for f in feature_names if f != feature_x]
                                feature_y = st.selectbox(
                                    "Select Y-axis feature",
                                    options=remaining_features,
                                    key="feature_y"
                                )
                            
                            if feature_x and feature_y and pd.api.types.is_numeric_dtype(df[feature_x]) and pd.api.types.is_numeric_dtype(df[feature_y]):
                                # Create scatter plot
                                fig = px.scatter(
                                    df,
                                    x=feature_x,
                                    y=feature_y,
                                    color=model_info["target"],
                                    title=f"Relationship between {feature_x} and {feature_y} by {model_info['target']}",
                                    opacity=0.7
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.header("Learning Curves")
            
            st.markdown("""
            Visualize how model performance changes with training data size to identify underfitting or overfitting issues.
            Learning curves show training and validation scores across different training set sizes.
            """)
            
            # Model selection for learning curves
            model_options = list(models.keys())
            selected_model = st.selectbox(
                "Select model to analyze",
                options=model_options,
                key="learning_curves_model"
            )
            
            if selected_model:
                model_info = models[selected_model]
                model_type = model_info["type"]
                
                st.info("""
                Generating learning curves requires retraining the model multiple times with different data sizes.
                This may take some time depending on your model complexity.
                """)
                
                # Options for learning curve
                col1, col2 = st.columns(2)
                
                with col1:
                    cv_folds = st.slider(
                        "Cross-validation folds",
                        min_value=2,
                        max_value=10,
                        value=5,
                        step=1
                    )
                
                with col2:
                    n_jobs = st.slider(
                        "Parallel jobs",
                        min_value=1,
                        max_value=4,
                        value=1,
                        step=1
                    )
                
                # Generate button
                if st.button("Generate Learning Curves"):
                    try:
                        with st.spinner("Generating learning curves... This may take a moment."):
                            # Prepare data
                            X = df[model_info["features"]]
                            if model_info["target"] in df.columns:
                                y = df[model_info["target"]]
                                
                                # Define scoring metric based on model type
                                scoring = 'accuracy' if model_type == 'classification' else 'neg_mean_squared_error'
                                
                                # Generate learning curve data
                                train_sizes = np.linspace(0.1, 1.0, 10)
                                
                                # Extract the final estimator from the pipeline
                                pipeline = model_info["pipeline"]
                                
                                # Generate the learning curve
                                train_sizes, train_scores, validation_scores = learning_curve(
                                    pipeline,
                                    X, y,
                                    train_sizes=train_sizes,
                                    cv=cv_folds,
                                    scoring=scoring,
                                    n_jobs=n_jobs
                                )
                                
                                # Calculate mean and std of training scores
                                train_mean = np.mean(train_scores, axis=1)
                                train_std = np.std(train_scores, axis=1)
                                validation_mean = np.mean(validation_scores, axis=1)
                                validation_std = np.std(validation_scores, axis=1)
                                
                                # Create figure
                                fig = go.Figure()
                                
                                # Add traces for training score
                                fig.add_trace(go.Scatter(
                                    x=train_sizes,
                                    y=train_mean,
                                    mode='lines+markers',
                                    name='Training score',
                                    line=dict(color='blue')
                                ))
                                
                                # Add confidence interval for training score
                                fig.add_trace(go.Scatter(
                                    x=np.concatenate([train_sizes, train_sizes[::-1]]),
                                    y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
                                    fill='toself',
                                    fillcolor='rgba(0, 0, 255, 0.1)',
                                    line=dict(color='rgba(0, 0, 255, 0)'),
                                    showlegend=False,
                                    name='Training score 95% CI'
                                ))
                                
                                # Add traces for validation score
                                fig.add_trace(go.Scatter(
                                    x=train_sizes,
                                    y=validation_mean,
                                    mode='lines+markers',
                                    name='Validation score',
                                    line=dict(color='red')
                                ))
                                
                                # Add confidence interval for validation score
                                fig.add_trace(go.Scatter(
                                    x=np.concatenate([train_sizes, train_sizes[::-1]]),
                                    y=np.concatenate([validation_mean + validation_std, (validation_mean - validation_std)[::-1]]),
                                    fill='toself',
                                    fillcolor='rgba(255, 0, 0, 0.1)',
                                    line=dict(color='rgba(255, 0, 0, 0)'),
                                    showlegend=False,
                                    name='Validation score 95% CI'
                                ))
                                
                                # Update layout
                                if model_type == 'classification':
                                    y_axis_title = 'Accuracy'
                                    title = f"Learning Curves for {selected_model} (Accuracy)"
                                else:
                                    y_axis_title = 'Negative MSE'
                                    title = f"Learning Curves for {selected_model} (Negative MSE)"
                                
                                fig.update_layout(
                                    title=title,
                                    xaxis_title='Training examples',
                                    yaxis_title=y_axis_title,
                                    legend=dict(
                                        x=0.01,
                                        y=0.99,
                                        bordercolor="Black",
                                        borderwidth=1
                                    )
                                )
                                
                                # Display the figure
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Interpretation guidance
                                st.subheader("Interpretation")
                                
                                # Basic interpretation
                                if model_type == 'classification':
                                    basic_interp = """
                                    ### What do these curves show?
                                    - **Training score** (blue): How well the model performs on its training data
                                    - **Validation score** (red): How well the model generalizes to unseen data
                                    
                                    ### Key insights:
                                    """
                                    
                                    # Analyze the curves
                                    train_final = train_mean[-1]
                                    val_final = validation_mean[-1]
                                    gap = train_final - val_final
                                    
                                    insights = []
                                    
                                    if gap > 0.2:
                                        insights.append("- **Overfitting detected**: There's a large gap between training and validation scores, suggesting the model memorizes training data but doesn't generalize well.")
                                    elif val_final < 0.6:
                                        insights.append("- **Underfitting detected**: Both curves show low scores, suggesting the model is too simple to capture the patterns in the data.")
                                    else:
                                        insights.append("- **Good fit**: The gap between training and validation scores is reasonable, suggesting good generalization.")
                                    
                                    if validation_mean[-1] > validation_mean[-2] and validation_mean[-2] > validation_mean[-3]:
                                        insights.append("- **More data helps**: Validation score is still improving with more data, suggesting that collecting more samples could improve performance.")
                                    else:
                                        insights.append("- **Diminishing returns**: Adding more data doesn't significantly improve validation performance, suggesting you've reached the model's capacity.")
                                    
                                    st.markdown(basic_interp + "\n".join(insights))
                                else:
                                    basic_interp = """
                                    ### What do these curves show?
                                    - **Training score** (blue): How well the model performs on its training data
                                    - **Validation score** (red): How well the model generalizes to unseen data
                                    - (Note: For regression, negative MSE is shown where higher values are better)
                                    
                                    ### Key insights:
                                    """
                                    
                                    # Analyze the curves
                                    train_final = train_mean[-1]
                                    val_final = validation_mean[-1]
                                    gap = train_final - val_final
                                    
                                    insights = []
                                    
                                    if gap > abs(val_final * 0.3):
                                        insights.append("- **Overfitting detected**: There's a large gap between training and validation errors, suggesting the model memorizes training data but doesn't generalize well.")
                                    elif val_final > -1.0 and train_final > -1.0:
                                        insights.append("- **Underfitting detected**: Both curves show high error (low negative MSE), suggesting the model is too simple to capture the patterns in the data.")
                                    else:
                                        insights.append("- **Good fit**: The gap between training and validation errors is reasonable, suggesting good generalization.")
                                    
                                    if validation_mean[-1] > validation_mean[-2] and validation_mean[-2] > validation_mean[-3]:
                                        insights.append("- **More data helps**: Validation error is still decreasing (score improving) with more data, suggesting that collecting more samples could improve performance.")
                                    else:
                                        insights.append("- **Diminishing returns**: Adding more data doesn't significantly reduce validation error, suggesting you've reached the model's capacity.")
                                    
                                    st.markdown(basic_interp + "\n".join(insights))
                                
                                # Recommendations
                                st.subheader("Recommendations")
                                
                                if model_type == 'classification':
                                    if gap > 0.2:
                                        st.markdown("""
                                        To address overfitting:
                                        - Simplify your model (reduce complexity)
                                        - Add regularization
                                        - Use more training data
                                        - Try feature selection to reduce dimensionality
                                        """)
                                    elif val_final < 0.6:
                                        st.markdown("""
                                        To address underfitting:
                                        - Use a more complex model
                                        - Add more relevant features
                                        - Reduce regularization
                                        - Try polynomial features or feature engineering
                                        """)
                                    else:
                                        st.markdown("""
                                        Your model shows a good balance between bias and variance. Consider:
                                        - Fine-tuning hyperparameters
                                        - Ensemble methods for potential improvement
                                        - Deploying the model as-is if performance meets requirements
                                        """)
                                else:
                                    if gap > abs(val_final * 0.3):
                                        st.markdown("""
                                        To address overfitting:
                                        - Simplify your regression model
                                        - Add regularization (L1, L2, or Elastic Net)
                                        - Use more training data
                                        - Try feature selection to reduce dimensionality
                                        """)
                                    elif val_final > -1.0 and train_final > -1.0:
                                        st.markdown("""
                                        To address underfitting:
                                        - Use a more complex regression model
                                        - Add more relevant features
                                        - Add polynomial features or interaction terms
                                        - Reduce regularization if using any
                                        """)
                                    else:
                                        st.markdown("""
                                        Your regression model shows a good balance between bias and variance. Consider:
                                        - Fine-tuning hyperparameters
                                        - Ensemble methods for potential improvement
                                        - Deploying the model as-is if performance meets requirements
                                        """)
                            
                            else:
                                st.error(f"Target variable '{model_info['target']}' not found in the dataset.")
                    
                    except Exception as e:
                        st.error(f"Error generating learning curves: {str(e)}")
                        st.info("This might be due to incompatibility with the model type or insufficient data.")

# Instructions when no data is loaded
else:
    show_file_required_warning()

# Footer
create_footer() 
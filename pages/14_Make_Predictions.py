import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import joblib
import io
import base64
from datetime import datetime
import os
import json

from utils import (
    load_css, 
    create_footer, 
    show_file_required_warning, 
    display_dataset_info
)

# Page config
st.set_page_config(page_title="Make Predictions", page_icon="🔮", layout="wide")
load_css()

# Main title
st.title("Make Predictions 🔮")
st.markdown("""
Use your trained models to make predictions on new data. Upload new data or enter values manually
to generate predictions using any model you've trained in the app.
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
        models["Logistic Regression"] = {
            "model": st.session_state.lr_trained_model,
            "pipeline": st.session_state.lr_pipeline,
            "features": st.session_state.lr_feature_names,
            "target": st.session_state.lr_target,
            "type": "classification",
            "classes": st.session_state.lr_classes if 'lr_classes' in st.session_state else None
        }
    
    if 'dt_trained_model' in st.session_state and st.session_state.dt_trained_model is not None:
        models["Decision Tree"] = {
            "model": st.session_state.dt_trained_model,
            "pipeline": st.session_state.dt_pipeline,
            "features": st.session_state.dt_feature_names,
            "target": st.session_state.dt_target,
            "type": "classification",
            "classes": st.session_state.dt_classes if 'dt_classes' in st.session_state else None
        }
    
    if 'knn_trained_model' in st.session_state and st.session_state.knn_trained_model is not None:
        models["K-Nearest Neighbors"] = {
            "model": st.session_state.knn_trained_model,
            "pipeline": st.session_state.knn_pipeline,
            "features": st.session_state.knn_feature_names,
            "target": st.session_state.knn_target,
            "type": "classification",
            "classes": st.session_state.knn_classes if 'knn_classes' in st.session_state else None
        }
    
    # Check for regression models
    if 'lr_trained_model' in st.session_state and st.session_state.lr_trained_model is not None:
        # Linear regression has same session state key as logistic regression
        # Check the model type to differentiate
        if hasattr(st.session_state.lr_trained_model, 'predict_proba'):
            # This is likely logistic regression, already added above
            pass
        else:
            # This is likely linear regression
            models["Linear Regression"] = {
                "model": st.session_state.lr_trained_model,
                "pipeline": st.session_state.lr_pipeline,
                "features": st.session_state.lr_feature_names,
                "target": st.session_state.lr_target,
                "type": "regression",
                "classes": None
            }
    
    if not models:
        st.warning("No trained models found. Please train at least one model first.")
    else:
        # Create tabs for prediction methods
        tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Upload File"])
        
        with tab1:
            st.header("Single Prediction")
            
            st.markdown("""
            Enter values for each feature to make a prediction using your trained model.
            """)
            
            # Model selection
            model_name = st.selectbox("Select Model", list(models.keys()))
            
            if model_name in models:
                selected_model = models[model_name]
                
                # Get model details
                model_type = selected_model["type"]
                features = selected_model["features"]
                target = selected_model["target"]
                classes = selected_model["classes"]
                
                st.subheader(f"Enter values for {model_name}")
                
                # Create input fields for features
                input_values = {}
                
                # Create columns for better UI
                num_cols = 3
                feature_groups = [features[i:i+num_cols] for i in range(0, len(features), num_cols)]
                
                for group in feature_groups:
                    cols = st.columns(num_cols)
                    for i, feature in enumerate(group):
                        # Get feature statistics from dataset to set defaults
                        feature_min = df[feature].min() if pd.api.types.is_numeric_dtype(df[feature]) else None
                        feature_max = df[feature].max() if pd.api.types.is_numeric_dtype(df[feature]) else None
                        feature_mean = df[feature].mean() if pd.api.types.is_numeric_dtype(df[feature]) else None
                        
                        # Create appropriate input field based on data type
                        with cols[i]:
                            if pd.api.types.is_numeric_dtype(df[feature]):
                                # Numeric input
                                input_values[feature] = st.number_input(
                                    f"{feature}",
                                    min_value=float(feature_min) if feature_min is not None else None,
                                    max_value=float(feature_max) if feature_max is not None else None,
                                    value=float(feature_mean) if feature_mean is not None else 0.0,
                                    step=0.1,
                                    key=f"input_{feature}"
                                )
                            elif pd.api.types.is_bool_dtype(df[feature]):
                                # Boolean input
                                input_values[feature] = st.checkbox(
                                    f"{feature}",
                                    value=df[feature].mode()[0] if not df[feature].empty else False,
                                    key=f"input_{feature}"
                                )
                            elif pd.api.types.is_categorical_dtype(df[feature]) or feature in df.select_dtypes(include=['object']).columns:
                                # Categorical input
                                unique_values = df[feature].dropna().unique().tolist()
                                default_value = df[feature].mode()[0] if not df[feature].empty else unique_values[0] if unique_values else ""
                                input_values[feature] = st.selectbox(
                                    f"{feature}",
                                    options=unique_values,
                                    index=unique_values.index(default_value) if default_value in unique_values else 0,
                                    key=f"input_{feature}"
                                )
                            else:
                                # Default to text input
                                input_values[feature] = st.text_input(
                                    f"{feature}",
                                    value=str(df[feature].mode()[0]) if not df[feature].empty else "",
                                    key=f"input_{feature}"
                                )
                
                # Make prediction button
                if st.button("Make Prediction", key="single_predict_button"):
                    try:
                        # Convert input values to DataFrame
                        input_df = pd.DataFrame([input_values])
                        
                        # Make prediction
                        if model_type == "classification":
                            prediction = selected_model["pipeline"].predict(input_df)
                            
                            # Get class label
                            if classes is not None:
                                prediction_label = classes[prediction[0]]
                            else:
                                prediction_label = str(prediction[0])
                            
                            # Try to get probability if available
                            try:
                                probabilities = selected_model["pipeline"].predict_proba(input_df)[0]
                                
                                # Display results with probabilities
                                st.success(f"Prediction: **{prediction_label}**")
                                
                                # Create a DataFrame for probabilities
                                if classes is not None:
                                    prob_df = pd.DataFrame({
                                        'Class': classes,
                                        'Probability': probabilities
                                    })
                                else:
                                    prob_df = pd.DataFrame({
                                        'Class': [f"Class {i}" for i in range(len(probabilities))],
                                        'Probability': probabilities
                                    })
                                
                                # Display probabilities
                                st.subheader("Class Probabilities")
                                st.dataframe(prob_df, use_container_width=True)
                                
                                # Plot probabilities
                                fig = px.bar(
                                    prob_df,
                                    x='Class',
                                    y='Probability',
                                    title="Class Probabilities",
                                    color='Probability',
                                    color_continuous_scale="Viridis"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                            except (AttributeError, NotImplementedError):
                                # Model doesn't support predict_proba
                                st.success(f"Prediction: **{prediction_label}**")
                        
                        else:  # Regression
                            prediction = selected_model["pipeline"].predict(input_df)[0]
                            
                            # Display result
                            st.success(f"Predicted {target}: **{prediction:.4f}**")
                            
                            # Show a gauge chart for the prediction
                            # Get target min and max from the original dataset
                            target_min = df[target].min()
                            target_max = df[target].max()
                            target_mean = df[target].mean()
                            
                            # Create gauge chart
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=prediction,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': f"Predicted {target}"},
                                gauge={
                                    'axis': {'range': [target_min, target_max]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [target_min, target_mean], 'color': "lightgray"},
                                        {'range': [target_mean, target_max], 'color': "gray"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': prediction
                                    }
                                }
                            ))
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
        
        with tab2:
            st.header("Batch Prediction")
            
            st.markdown("""
            Generate predictions for multiple data points at once. Enter values in the table below
            and click 'Make Predictions' to get results for all rows.
            """)
            
            # Model selection
            model_name = st.selectbox("Select Model", list(models.keys()), key="batch_model_select")
            
            if model_name in models:
                selected_model = models[model_name]
                
                # Get model details
                model_type = selected_model["type"]
                features = selected_model["features"]
                target = selected_model["target"]
                classes = selected_model["classes"]
                
                st.subheader("Enter data for batch prediction")
                
                # Number of rows to predict
                num_rows = st.number_input(
                    "Number of rows to predict",
                    min_value=1,
                    max_value=100,
                    value=5,
                    step=1
                )
                
                # Create empty dataframe with features as columns
                batch_df = pd.DataFrame(columns=features)
                
                # Add empty rows
                for _ in range(num_rows):
                    batch_df = pd.concat([batch_df, pd.DataFrame({col: [''] for col in features})], ignore_index=True)
                
                # Convert appropriate columns to numeric
                for col in features:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        batch_df[col] = pd.to_numeric(batch_df[col], errors='coerce')
                
                # Allow user to edit the DataFrame
                edited_df = st.data_editor(
                    batch_df,
                    use_container_width=True,
                    num_rows="fixed",
                    hide_index=True
                )
                
                # Make predictions button
                if st.button("Make Predictions", key="batch_predict_button"):
                    try:
                        # Validate input data
                        has_empty = False
                        for col in features:
                            if edited_df[col].isnull().any() or (edited_df[col] == '').any():
                                has_empty = True
                                break
                        
                        if has_empty:
                            st.warning("Please fill in all values before making predictions.")
                        else:
                            # Convert data types if needed
                            for col in features:
                                if pd.api.types.is_numeric_dtype(df[col]):
                                    edited_df[col] = pd.to_numeric(edited_df[col], errors='coerce')
                            
                            # Make predictions
                            if model_type == "classification":
                                predictions = selected_model["pipeline"].predict(edited_df)
                                
                                # Get class labels
                                if classes is not None:
                                    prediction_labels = [classes[p] for p in predictions]
                                else:
                                    prediction_labels = [str(p) for p in predictions]
                                
                                # Add predictions to dataframe
                                result_df = edited_df.copy()
                                result_df[f"Predicted {target}"] = prediction_labels
                                
                                # Try to get probabilities if available
                                try:
                                    probabilities = selected_model["pipeline"].predict_proba(edited_df)
                                    
                                    # If classes are available, add probability columns
                                    if classes is not None:
                                        for i, cls in enumerate(classes):
                                            result_df[f"Prob: {cls}"] = probabilities[:, i]
                                    else:
                                        for i in range(probabilities.shape[1]):
                                            result_df[f"Prob: Class {i}"] = probabilities[:, i]
                                
                                except (AttributeError, NotImplementedError):
                                    # Model doesn't support predict_proba
                                    pass
                            
                            else:  # Regression
                                predictions = selected_model["pipeline"].predict(edited_df)
                                
                                # Add predictions to dataframe
                                result_df = edited_df.copy()
                                result_df[f"Predicted {target}"] = predictions
                            
                            # Display results
                            st.subheader("Prediction Results")
                            st.dataframe(result_df, use_container_width=True)
                            
                            # Create download link for results
                            csv = result_df.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="batch_predictions.csv">Download Prediction Results (CSV)</a>'
                            st.markdown(href, unsafe_allow_html=True)
                            
                            # Visualize results
                            st.subheader("Results Visualization")
                            
                            if model_type == "classification":
                                # Count predictions by class
                                pred_counts = pd.Series(prediction_labels).value_counts().reset_index()
                                pred_counts.columns = ['Class', 'Count']
                                
                                # Create bar chart
                                fig = px.bar(
                                    pred_counts,
                                    x='Class',
                                    y='Count',
                                    title="Distribution of Predictions",
                                    color='Class'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            else:  # Regression
                                # Histogram of predictions
                                fig = px.histogram(
                                    result_df,
                                    x=f"Predicted {target}",
                                    title=f"Distribution of Predicted {target}",
                                    nbins=20
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error making batch predictions: {str(e)}")
        
        with tab3:
            st.header("Upload File for Predictions")
            
            st.markdown("""
            Upload a CSV file with feature data to make predictions on multiple samples.
            The file should contain columns for all the features required by the selected model.
            """)
            
            # Model selection
            model_name = st.selectbox("Select Model", list(models.keys()), key="file_model_select")
            
            if model_name in models:
                selected_model = models[model_name]
                
                # Get model details
                model_type = selected_model["type"]
                features = selected_model["features"]
                target = selected_model["target"]
                classes = selected_model["classes"]
                
                # File uploader
                uploaded_file = st.file_uploader(
                    "Upload CSV file with feature data",
                    type=["csv"],
                    key="prediction_file_upload"
                )
                
                if uploaded_file is not None:
                    try:
                        # Read file
                        upload_df = pd.read_csv(uploaded_file)
                        
                        # Display preview
                        st.subheader("Data Preview")
                        st.dataframe(upload_df.head(), use_container_width=True)
                        
                        # Check if all required features are present
                        missing_features = [f for f in features if f not in upload_df.columns]
                        
                        if missing_features:
                            st.error(f"Missing required features: {', '.join(missing_features)}")
                            st.info(f"The selected model requires these features: {', '.join(features)}")
                        else:
                            # Make predictions button
                            if st.button("Make Predictions", key="file_predict_button"):
                                try:
                                    # Extract feature columns
                                    pred_df = upload_df[features].copy()
                                    
                                    # Make predictions
                                    if model_type == "classification":
                                        predictions = selected_model["pipeline"].predict(pred_df)
                                        
                                        # Get class labels
                                        if classes is not None:
                                            prediction_labels = [classes[p] for p in predictions]
                                        else:
                                            prediction_labels = [str(p) for p in predictions]
                                        
                                        # Add predictions to dataframe
                                        result_df = upload_df.copy()
                                        result_df[f"Predicted {target}"] = prediction_labels
                                        
                                        # Try to get probabilities if available
                                        try:
                                            probabilities = selected_model["pipeline"].predict_proba(pred_df)
                                            
                                            # If classes are available, add probability columns
                                            if classes is not None:
                                                for i, cls in enumerate(classes):
                                                    result_df[f"Prob: {cls}"] = probabilities[:, i]
                                            else:
                                                for i in range(probabilities.shape[1]):
                                                    result_df[f"Prob: Class {i}"] = probabilities[:, i]
                                        
                                        except (AttributeError, NotImplementedError):
                                            # Model doesn't support predict_proba
                                            pass
                                    
                                    else:  # Regression
                                        predictions = selected_model["pipeline"].predict(pred_df)
                                        
                                        # Add predictions to dataframe
                                        result_df = upload_df.copy()
                                        result_df[f"Predicted {target}"] = predictions
                                    
                                    # Display results
                                    st.subheader("Prediction Results")
                                    
                                    # Calculate number of rows to display
                                    num_display_rows = min(100, len(result_df))
                                    st.dataframe(result_df.head(num_display_rows), use_container_width=True)
                                    
                                    if len(result_df) > num_display_rows:
                                        st.info(f"Showing first {num_display_rows} of {len(result_df)} rows. Download the full results using the link below.")
                                    
                                    # Create download link for results
                                    csv = result_df.to_csv(index=False)
                                    b64 = base64.b64encode(csv.encode()).decode()
                                    href = f'<a href="data:file/csv;base64,{b64}" download="file_predictions.csv">Download Prediction Results (CSV)</a>'
                                    st.markdown(href, unsafe_allow_html=True)
                                    
                                    # Visualize results
                                    st.subheader("Results Visualization")
                                    
                                    if model_type == "classification":
                                        # Count predictions by class
                                        pred_counts = pd.Series(prediction_labels).value_counts().reset_index()
                                        pred_counts.columns = ['Class', 'Count']
                                        
                                        # Create bar chart
                                        fig = px.bar(
                                            pred_counts,
                                            x='Class',
                                            y='Count',
                                            title="Distribution of Predictions",
                                            color='Class'
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Create pie chart
                                        fig2 = px.pie(
                                            pred_counts,
                                            values='Count',
                                            names='Class',
                                            title="Proportion of Predictions by Class"
                                        )
                                        
                                        st.plotly_chart(fig2, use_container_width=True)
                                    
                                    else:  # Regression
                                        # Histogram of predictions
                                        fig = px.histogram(
                                            result_df,
                                            x=f"Predicted {target}",
                                            title=f"Distribution of Predicted {target}",
                                            nbins=20
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Statistics of predictions
                                        pred_stats = result_df[f"Predicted {target}"].describe().reset_index()
                                        pred_stats.columns = ['Statistic', 'Value']
                                        
                                        st.subheader("Prediction Statistics")
                                        st.dataframe(pred_stats, use_container_width=True)
                                
                                except Exception as e:
                                    st.error(f"Error making predictions: {str(e)}")
                    
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
                        st.info("Please ensure your CSV file is properly formatted.")

# Instructions when no data is loaded
else:
    show_file_required_warning()

# Footer
create_footer() 
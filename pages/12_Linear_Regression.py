import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import time
import io
import pickle
import base64
import warnings

from utils import (
    load_css, 
    create_footer, 
    show_file_required_warning, 
    display_dataset_info,
    get_numeric_columns,
    get_categorical_columns
)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Page config
st.set_page_config(page_title="Linear Regression", page_icon="📈", layout="wide")
load_css()

# Main title
st.title("Linear Regression 📈")
st.markdown("""
Train a Linear Regression model to predict continuous values. This page helps you set up, train, 
evaluate, and visualize a Linear Regression model for your regression problems.
""")

# Check if data is loaded
df = st.session_state.get('df', None)
if df is not None:
    # Display dataset info
    display_dataset_info()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Train Model", "Evaluate Model", "Make Predictions"])
    
    with tab1:
        st.header("Linear Regression Overview")
        
        st.markdown("""
        ### What is Linear Regression?
        
        Linear Regression is a fundamental supervised learning algorithm that models the relationship between a dependent variable (target) 
        and one or more independent variables (features) by fitting a linear equation to the observed data.
        
        ### When to Use Linear Regression
        
        Linear Regression works well when:
        - You need to predict a continuous value (e.g., price, temperature, sales)
        - The relationship between features and target is approximately linear
        - You want a model that's easy to interpret and explain
        - Feature interactions are minimal or can be explicitly modeled
        
        ### Advantages of Linear Regression
        
        - **Simplicity**: Easy to understand and implement
        - **Interpretability**: Coefficients directly represent feature importance and direction
        - **Efficiency**: Fast to train, even on large datasets
        - **Performance**: Works well when the relationship is truly linear
        - **Statistical properties**: Provides confidence intervals and hypothesis tests for parameters
        
        ### Limitations
        
        - Assumes a linear relationship between features and target
        - Sensitive to outliers in the data
        - Can underfit complex, non-linear relationships
        - Assumes independence of features (multicollinearity can be problematic)
        - Assumes constant variance of errors (homoscedasticity)
        """)
        
        # Show dataset preview
        st.subheader("Dataset Preview")
        st.dataframe(df.head(5), use_container_width=True)
        
        # Check for numeric target variables
        numeric_columns = get_numeric_columns(df)
        if numeric_columns:
            st.subheader("Potential Target Variables")
            st.markdown("These numeric columns could be suitable targets for regression:")
            
            # Show potential target variables with distribution
            for col in numeric_columns[:5]:  # Limit to first 5 to avoid cluttering the UI
                try:
                    # Calculate basic statistics
                    stats = df[col].describe()
                    
                    # Show statistics
                    st.markdown(f"**{col}**")
                    stats_cols = st.columns(4)
                    with stats_cols[0]:
                        st.metric("Mean", f"{stats['mean']:.2f}")
                    with stats_cols[1]:
                        st.metric("Standard Deviation", f"{stats['std']:.2f}")
                    with stats_cols[2]:
                        st.metric("Min", f"{stats['min']:.2f}")
                    with stats_cols[3]:
                        st.metric("Max", f"{stats['max']:.2f}")
                    
                    # Create distribution plot
                    fig = px.histogram(
                        df, 
                        x=col,
                        title=f"Distribution of {col}",
                        marginal="box"  # Add a box plot on the marginal
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not analyze column {col}: {str(e)}")
        else:
            st.info("No numeric columns detected in this dataset. To use Linear Regression, you need a numeric target variable.")
            
    with tab2:
        st.header("Train Linear Regression Model")
        
        # Initialize session state variables for model training
        if 'lr_target' not in st.session_state:
            st.session_state.lr_target = None
        if 'lr_features' not in st.session_state:
            st.session_state.lr_features = []
        if 'lr_trained_model' not in st.session_state:
            st.session_state.lr_trained_model = None
        if 'lr_model_metrics' not in st.session_state:
            st.session_state.lr_model_metrics = None
        if 'lr_pipeline' not in st.session_state:
            st.session_state.lr_pipeline = None
        if 'lr_feature_names' not in st.session_state:
            st.session_state.lr_feature_names = None
        if 'lr_X_train' not in st.session_state:
            st.session_state.lr_X_train = None
        if 'lr_X_test' not in st.session_state:
            st.session_state.lr_X_test = None
        if 'lr_y_train' not in st.session_state:
            st.session_state.lr_y_train = None
        if 'lr_y_test' not in st.session_state:
            st.session_state.lr_y_test = None
        if 'lr_coefficients' not in st.session_state:
            st.session_state.lr_coefficients = None
        
        # Step 1: Select target variable
        st.subheader("Step 1: Select Target Variable")
        
        # Get suitable numeric columns for regression
        suitable_targets = []
        for col in numeric_columns:
            # Check if the column has enough unique values to be a good regression target
            if df[col].nunique() > 10:  # Arbitrary threshold for regression
                suitable_targets.append(col)
        
        if not suitable_targets:
            st.warning("No suitable numeric target variables found. Ideal target variables should be continuous.")
            suitable_targets = numeric_columns  # Use all numeric columns as fallback
        
        target_col = st.selectbox(
            "Select target variable (what you want to predict):",
            options=suitable_targets,
            index=0 if suitable_targets else None,
            help="The column containing the continuous values you want to predict"
        )
        
        if target_col:
            st.session_state.lr_target = target_col
            
            # Show distribution of target variable
            fig = px.histogram(
                df, 
                x=target_col,
                title=f"Distribution of {target_col}",
                marginal="box"  # Add a box plot on the marginal
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Step 2: Select features
            st.subheader("Step 2: Select Features")
            
            # Get all columns except target
            potential_features = [col for col in df.columns if col != target_col]
            
            # Allow selecting all or specific features
            select_all = st.checkbox("Select all features", value=False)
            
            if select_all:
                selected_features = potential_features
            else:
                selected_features = st.multiselect(
                    "Select features to include in the model:",
                    options=potential_features,
                    default=st.session_state.lr_features if st.session_state.lr_features else []
                )
            
            # Update session state
            st.session_state.lr_features = selected_features
            
            if not selected_features:
                st.warning("Please select at least one feature for training.")
            
            # Step 3: Configure model parameters
            if selected_features:
                st.subheader("Step 3: Configure Model Parameters")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Linear regression specific parameters
                    fit_intercept = st.checkbox(
                        "Fit intercept",
                        value=True,
                        help="Whether to calculate the intercept for the model. If False, no intercept will be used."
                    )
                    
                    normalize = st.checkbox(
                        "Normalize",
                        value=False,
                        help="If True, the regressors will be normalized before regression by subtracting the mean and dividing by the l2-norm."
                    )
                    
                with col2:
                    # Data preprocessing options
                    test_size = st.slider(
                        "Test set size:",
                        min_value=0.1,
                        max_value=0.5,
                        value=0.2,
                        step=0.05,
                        help="Proportion of data to use for testing the model."
                    )
                    
                    random_state = st.number_input(
                        "Random seed:",
                        min_value=0,
                        max_value=1000,
                        value=42,
                        help="Seed for reproducible results."
                    )
                    
                    apply_scaling = st.checkbox(
                        "Apply feature scaling",
                        value=True,
                        help="Standardize features for better model performance (recommended)"
                    )
                
                # Step 4: Train the model
                st.subheader("Step 4: Train the Model")
                
                train_button = st.button("Train Linear Regression Model")
                
                if train_button:
                    if not selected_features:
                        st.error("Please select at least one feature for training.")
                    else:
                        # Show progress
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        try:
                            status_text.text("Preparing data...")
                            
                            # Prepare data
                            X = df[selected_features].copy()
                            y = df[target_col].copy()
                            
                            # Handle missing values
                            if X.isnull().any().any() or y.isnull().any():
                                status_text.text("Handling missing values...")
                                # Drop rows with missing target values
                                rows_before = len(y)
                                missing_y_idx = y.isnull()
                                if missing_y_idx.any():
                                    y = y[~missing_y_idx]
                                    X = X[~missing_y_idx]
                                rows_after = len(y)
                                if rows_before > rows_after:
                                    st.info(f"Dropped {rows_before - rows_after} rows with missing target values.")
                            
                            # Check if there's enough data after dropping nulls
                            if len(X) < 20:
                                st.error("Not enough data after removing missing values. Need at least 20 samples.")
                                st.stop()
                            
                            progress_bar.progress(10)
                            
                            # Identify numeric and categorical features
                            numeric_features = get_numeric_columns(X)
                            categorical_features = [col for col in X.columns if col not in numeric_features]
                            
                            # Create preprocessing pipeline
                            preprocessor_parts = []
                            
                            # For numeric features
                            if numeric_features:
                                numeric_steps = [('imputer', SimpleImputer(strategy='median'))]
                                if apply_scaling:
                                    numeric_steps.append(('scaler', StandardScaler()))
                                
                                numeric_transformer = Pipeline(steps=numeric_steps)
                                preprocessor_parts.append(('num', numeric_transformer, numeric_features))
                            
                            # For categorical features
                            if categorical_features:
                                categorical_transformer = Pipeline(steps=[
                                    ('imputer', SimpleImputer(strategy='most_frequent')),
                                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                                ])
                                preprocessor_parts.append(('cat', categorical_transformer, categorical_features))
                            
                            # Create column transformer
                            preprocessor = ColumnTransformer(
                                transformers=preprocessor_parts,
                                remainder='drop'
                            )
                            
                            progress_bar.progress(20)
                            status_text.text("Building model pipeline...")
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, 
                                test_size=test_size, 
                                random_state=int(random_state)
                            )
                            
                            progress_bar.progress(30)
                            status_text.text("Training model...")
                            
                            # Create model with selected parameters
                            model = LinearRegression(
                                fit_intercept=fit_intercept,
                                normalize=normalize,
                                n_jobs=-1  # Use all processors
                            )
                            
                            # Create full pipeline
                            pipeline = Pipeline(steps=[
                                ('preprocessor', preprocessor),
                                ('regressor', model)
                            ])
                            
                            # Train model
                            start_time = time.time()
                            pipeline.fit(X_train, y_train)
                            training_time = time.time() - start_time
                            
                            progress_bar.progress(80)
                            status_text.text("Evaluating model...")
                            
                            # Make predictions
                            y_pred = pipeline.predict(X_test)
                            
                            # Calculate metrics
                            r2 = r2_score(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, y_pred)
                            
                            # Calculate additional metrics
                            explained_variance = 1 - (np.var(y_test - y_pred) / np.var(y_test))
                            mean_abs_percentage_error = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                            
                            # Create metrics dictionary
                            metrics = {
                                "R² Score": r2,
                                "Mean Squared Error": mse,
                                "Root Mean Squared Error": rmse,
                                "Mean Absolute Error": mae,
                                "Explained Variance": explained_variance,
                                "Mean Absolute Percentage Error": mean_abs_percentage_error,
                                "Training Time": training_time,
                                "Test Set Size": len(X_test)
                            }
                            
                            # Extract model coefficients
                            model_coefficients = None
                            try:
                                if hasattr(pipeline, 'named_steps') and 'regressor' in pipeline.named_steps:
                                    regressor = pipeline.named_steps['regressor']
                                    if hasattr(regressor, 'coef_'):
                                        raw_coefficients = regressor.coef_
                                        
                                        # If we have a preprocessor, we need to get the feature names
                                        feature_names_out = []
                                        
                                        if 'preprocessor' in pipeline.named_steps:
                                            preprocessor = pipeline.named_steps['preprocessor']
                                            if hasattr(preprocessor, 'transformers_'):
                                                for name, transformer, features in preprocessor.transformers_:
                                                    if name == 'cat' and hasattr(transformer, 'named_steps') and 'onehot' in transformer.named_steps:
                                                        onehot = transformer.named_steps['onehot']
                                                        if hasattr(onehot, 'get_feature_names_out'):
                                                            cat_features = onehot.get_feature_names_out(features)
                                                            feature_names_out.extend(cat_features)
                                                        else:
                                                            feature_names_out.extend([f"{f}_encoded" for f in features])
                                                    else:
                                                        feature_names_out.extend(features)
                                                    
                                        # If we got the feature names, create a dictionary
                                        if len(feature_names_out) == len(raw_coefficients):
                                            model_coefficients = dict(zip(feature_names_out, raw_coefficients))
                                        else:
                                            # Fall back to generic feature names
                                            model_coefficients = dict(zip([f"Feature_{i}" for i in range(len(raw_coefficients))], raw_coefficients))
                                        
                                        # Add intercept if available
                                        if hasattr(regressor, 'intercept_'):
                                            model_coefficients['Intercept'] = regressor.intercept_
                            except Exception as e:
                                st.warning(f"Could not extract model coefficients: {str(e)}")
                            
                            progress_bar.progress(90)
                            
                            # Save to session state
                            st.session_state.lr_trained_model = model
                            st.session_state.lr_model_metrics = metrics
                            st.session_state.lr_pipeline = pipeline
                            st.session_state.lr_feature_names = selected_features
                            st.session_state.lr_X_train = X_train
                            st.session_state.lr_X_test = X_test
                            st.session_state.lr_y_train = y_train
                            st.session_state.lr_y_test = y_test
                            st.session_state.lr_coefficients = model_coefficients
                            
                            progress_bar.progress(100)
                            status_text.text("Training complete!")
                            
                            # Display results
                            st.success("✅ Model training complete!")
                            
                            # Display metrics in a nice format
                            st.subheader("Model Performance Metrics")
                            
                            # Create metrics in a grid
                            metric_cols1 = st.columns(4)
                            
                            with metric_cols1[0]:
                                st.metric("R² Score", f"{metrics['R² Score']:.4f}")
                                st.caption("Proportion of variance explained (0-1, higher is better)")
                                
                            with metric_cols1[1]:
                                st.metric("RMSE", f"{metrics['Root Mean Squared Error']:.4f}")
                                st.caption("Root mean squared error (lower is better)")
                                
                            with metric_cols1[2]:
                                st.metric("MAE", f"{metrics['Mean Absolute Error']:.4f}")
                                st.caption("Mean absolute error (lower is better)")
                                
                            with metric_cols1[3]:
                                st.metric("MAPE", f"{metrics['Mean Absolute Percentage Error']:.2f}%")
                                st.caption("Mean absolute percentage error")
                            
                            # Create a scatter plot of actual vs predicted values
                            st.subheader("Actual vs. Predicted Values")
                            
                            pred_df = pd.DataFrame({
                                'Actual': y_test,
                                'Predicted': y_pred,
                                'Residuals': y_test - y_pred
                            })
                            
                            fig = px.scatter(
                                pred_df,
                                x='Actual',
                                y='Predicted',
                                title='Actual vs. Predicted Values',
                                labels={'Actual': f'Actual {target_col}', 'Predicted': f'Predicted {target_col}'},
                                opacity=0.7
                            )
                            
                            # Add perfect prediction line
                            min_val = min(pred_df['Actual'].min(), pred_df['Predicted'].min())
                            max_val = max(pred_df['Actual'].max(), pred_df['Predicted'].max())
                            fig.add_trace(
                                go.Scatter(
                                    x=[min_val, max_val],
                                    y=[min_val, max_val],
                                    mode='lines',
                                    name='Perfect Prediction',
                                    line=dict(color='red', dash='dash')
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display residuals plot
                            st.subheader("Residuals Analysis")
                            
                            fig = px.scatter(
                                pred_df,
                                x='Predicted',
                                y='Residuals',
                                title='Residuals vs. Predicted Values',
                                labels={'Predicted': f'Predicted {target_col}', 'Residuals': 'Residuals'},
                                opacity=0.7
                            )
                            
                            # Add horizontal line at y=0
                            fig.add_hline(
                                y=0,
                                line_dash="dash",
                                line_color="red",
                                annotation_text="Zero Line",
                                annotation_position="bottom right"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display histogram of residuals
                            fig = px.histogram(
                                pred_df,
                                x='Residuals',
                                title='Distribution of Residuals',
                                marginal='box'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display model coefficients if available
                            if model_coefficients:
                                st.subheader("Model Coefficients")
                                
                                # Create a dataframe from the coefficients
                                coef_items = list(model_coefficients.items())
                                # Move intercept to the end if it exists
                                if 'Intercept' in model_coefficients:
                                    intercept_value = model_coefficients['Intercept']
                                    coef_items = [(k, v) for k, v in coef_items if k != 'Intercept']
                                    coef_items.append(('Intercept', intercept_value))
                                
                                coef_df = pd.DataFrame(coef_items, columns=['Feature', 'Coefficient'])
                                coef_df = coef_df.sort_values('Coefficient', ascending=False)
                                
                                # Create a bar chart of coefficients
                                fig = px.bar(
                                    coef_df,
                                    x='Feature',
                                    y='Coefficient',
                                    title='Feature Coefficients',
                                    labels={'Feature': 'Feature', 'Coefficient': 'Coefficient Value'},
                                    color='Coefficient',
                                    color_continuous_scale=px.colors.diverging.RdBu_r,
                                    color_continuous_midpoint=0
                                )
                                
                                fig.update_layout(xaxis_tickangle=-45)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Display coefficient values in a table
                                st.dataframe(coef_df, use_container_width=True)
                                
                                # Interpretation guide
                                st.info("""
                                **Interpreting Coefficients**: 
                                
                                - Positive coefficients indicate that as the feature increases, the target variable increases
                                - Negative coefficients indicate that as the feature increases, the target variable decreases
                                - The magnitude of the coefficient indicates the strength of the effect (when features are scaled)
                                - The intercept represents the predicted value when all features are zero
                                """)
                            
                            # Prompt to go to evaluation tab
                            st.info("👉 Go to the 'Evaluate Model' tab to explore model details and visualizations.")
                            
                            # Save model button
                            st.subheader("Save Trained Model")
                            
                            # Save model to download
                            def get_model_download_link(pipeline, filename='linear_regression_model.pkl'):
                                """Generate a download link for the trained model"""
                                buffer = io.BytesIO()
                                pickle.dump(pipeline, buffer)
                                buffer.seek(0)
                                b64 = base64.b64encode(buffer.read()).decode()
                                href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download Trained Model</a>'
                                return href
                            
                            st.markdown(get_model_download_link(pipeline), unsafe_allow_html=True)
                            st.caption("Download the trained model to use later or in other applications.")
                            
                        except Exception as e:
                            st.error(f"An error occurred during model training: {str(e)}")
                            st.info("Try selecting different features or adjusting model parameters.")
    
    with tab3:
        st.header("Evaluate Model")
        
        # Check if a model has been trained
        if st.session_state.get('lr_trained_model') is None:
            st.warning("⚠️ You need to train a model first. Go to the 'Train Model' tab to train a Linear Regression model.")
        else:
            # Get model and data from session state
            model = st.session_state.lr_trained_model
            pipeline = st.session_state.lr_pipeline
            X_test = st.session_state.lr_X_test
            y_test = st.session_state.lr_y_test
            X_train = st.session_state.lr_X_train
            y_train = st.session_state.lr_y_train
            feature_names = st.session_state.lr_feature_names
            metrics = st.session_state.lr_model_metrics
            coefficients = st.session_state.lr_coefficients
            target_col = st.session_state.lr_target
            
            # Create subtabs for different evaluation aspects
            eval_tab1, eval_tab2, eval_tab3 = st.tabs([
                "Model Performance", 
                "Residual Analysis", 
                "Feature Importance"
            ])
            
            with eval_tab1:
                st.subheader("Performance Metrics")
                
                # Display metrics in a nice format with descriptions
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("R² Score", f"{metrics['R² Score']:.4f}")
                    st.caption("Coefficient of determination (0-1, higher is better)")
                    st.caption("Proportion of variance in the target that is predictable from the features")
                    
                    st.metric("Mean Squared Error", f"{metrics['Mean Squared Error']:.4f}")
                    st.caption("Average of squared differences between predicted and actual values")
                    
                    st.metric("Root Mean Squared Error", f"{metrics['Root Mean Squared Error']:.4f}")
                    st.caption("Square root of MSE, interpretable in the same units as the target")
                
                with col2:
                    st.metric("Mean Absolute Error", f"{metrics['Mean Absolute Error']:.4f}")
                    st.caption("Average of absolute differences between predicted and actual values")
                    
                    st.metric("Explained Variance", f"{metrics['Explained Variance']:.4f}")
                    st.caption("Proportion of variance explained (similar to R²)")
                    
                    st.metric("Mean Absolute % Error", f"{metrics['Mean Absolute Percentage Error']:.2f}%")
                    st.caption("Average percentage difference between predicted and actual values")
                
                # Actual vs Predicted plot
                st.subheader("Actual vs. Predicted Values")
                
                # Make predictions
                y_pred_test = pipeline.predict(X_test)
                
                # Prepare data for plotting
                results_df = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': y_pred_test,
                    'Error': y_test - y_pred_test
                })
                
                # Create scatter plot
                fig = px.scatter(
                    results_df,
                    x='Actual',
                    y='Predicted',
                    title='Actual vs. Predicted Values (Test Set)',
                    labels={'Actual': f'Actual {target_col}', 'Predicted': f'Predicted {target_col}'},
                    opacity=0.7,
                    color='Error',
                    color_continuous_scale='RdBu_r',
                    color_continuous_midpoint=0
                )
                
                # Add perfect prediction line
                min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
                max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='black', dash='dash')
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add an explanation of the plot
                st.markdown("""
                **Interpretation:**
                - Points close to the dashed line indicate accurate predictions
                - Blue points show under-predictions (actual > predicted)
                - Red points show over-predictions (predicted > actual)
                - The spread of points indicates model error
                """)
                
                # Test vs Training Performance
                st.subheader("Test vs. Training Performance")
                
                # Get training predictions
                y_pred_train = pipeline.predict(X_train)
                
                # Calculate metrics on training set
                r2_train = r2_score(y_train, y_pred_train)
                rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
                mae_train = mean_absolute_error(y_train, y_pred_train)
                
                # Display comparison
                metrics_comp = pd.DataFrame({
                    'Metric': ['R² Score', 'RMSE', 'MAE'],
                    'Training': [r2_train, rmse_train, mae_train],
                    'Test': [metrics['R² Score'], metrics['Root Mean Squared Error'], metrics['Mean Absolute Error']]
                })
                
                # Create comparison bar chart
                fig = px.bar(
                    metrics_comp, 
                    x='Metric', 
                    y=['Training', 'Test'],
                    barmode='group',
                    title='Model Performance: Training vs. Test Set',
                    color_discrete_sequence=['#1f77b4', '#ff7f0e']
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation guidance
                if r2_train > metrics['R² Score'] * 1.2:  # If training R² is 20% better than test
                    st.warning("""
                    **Potential Overfitting Detected**: The model performs significantly better on the training set than 
                    on the test set. This may indicate that the model has memorized the training data rather than learning 
                    generalizable patterns.
                    
                    **Suggestions:**
                    - Reduce model complexity
                    - Gather more training data
                    - Use regularization techniques
                    """)
                else:
                    st.success("""
                    **Good Generalization**: The model performs similarly on both training and test sets, 
                    suggesting that it has learned general patterns rather than memorizing the training data.
                    """)
            
            with eval_tab2:
                st.subheader("Residual Analysis")
                
                # Create residuals
                residuals_df = pd.DataFrame({
                    'Predicted': y_pred_test,
                    'Residuals': y_test - y_pred_test,
                    'Standardized Residuals': (y_test - y_pred_test) / np.std(y_test - y_pred_test)
                })
                
                # Residuals vs Predicted plot
                st.subheader("Residuals vs. Predicted Values")
                
                fig = px.scatter(
                    residuals_df,
                    x='Predicted',
                    y='Residuals',
                    title='Residuals vs. Predicted Values',
                    labels={'Predicted': f'Predicted {target_col}', 'Residuals': 'Residuals'},
                    opacity=0.7
                )
                
                # Add horizontal line at y=0
                fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Zero Line",
                    annotation_position="bottom right"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Explanation for residual plot
                st.markdown("""
                **Interpretation of Residual Plot:**
                
                - **Random scatter around zero line**: Indicates a good fit with no systematic bias
                - **Pattern/trend in residuals**: Suggests model may be missing important features or relationships
                - **Fan shape**: Indicates heteroscedasticity (non-constant variance), which violates linear regression assumptions
                - **Outliers**: Points far from zero line may represent unusual cases worth investigating
                """)
                
                # Distribution of residuals
                st.subheader("Distribution of Residuals")
                
                # Plot histogram with normal curve overlay
                fig = px.histogram(
                    residuals_df,
                    x='Residuals',
                    title='Distribution of Residuals',
                    histnorm='probability density',
                    marginal='box'
                )
                
                # Add normal distribution curve
                mean = np.mean(residuals_df['Residuals'])
                std = np.std(residuals_df['Residuals'])
                x = np.linspace(mean - 4*std, mean + 4*std, 100)
                y = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * std**2))
                
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode='lines',
                        name='Normal Distribution',
                        line=dict(color='red')
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Normal Q-Q plot
                st.subheader("Normal Q-Q Plot")
                
                # Prepare Q-Q plot data
                from scipy import stats
                
                residuals = residuals_df['Standardized Residuals'].sort_values()
                n = len(residuals)
                theoretical_quantiles = np.array([stats.norm.ppf((i + 0.5) / n) for i in range(n)])
                
                # Create Q-Q plot
                fig = px.scatter(
                    x=theoretical_quantiles,
                    y=residuals,
                    labels={'x': 'Theoretical Quantiles', 'y': 'Standardized Residuals'},
                    title='Normal Q-Q Plot'
                )
                
                # Add reference line
                min_q = min(theoretical_quantiles)
                max_q = max(theoretical_quantiles)
                fig.add_trace(
                    go.Scatter(
                        x=[min_q, max_q],
                        y=[min_q, max_q],
                        mode='lines',
                        name='Reference Line',
                        line=dict(color='red', dash='dash')
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **Interpretation of Q-Q Plot:**
                
                - **Points following the reference line**: Residuals are normally distributed
                - **Deviation from line**: Non-normal distribution, may indicate issues with model assumptions
                - **S-shape curve**: Indicates skewness in residuals
                - **Heavy tails**: More extreme values than expected in a normal distribution
                """)
                
                # Additional residual statistics
                st.subheader("Residual Statistics")
                
                res_stats = pd.DataFrame({
                    'Statistic': [
                        'Mean of Residuals',
                        'Standard Deviation',
                        'Minimum',
                        'Maximum',
                        'Shapiro-Wilk p-value (normality test)'
                    ],
                    'Value': [
                        f"{np.mean(residuals_df['Residuals']):.6f}",
                        f"{np.std(residuals_df['Residuals']):.4f}",
                        f"{residuals_df['Residuals'].min():.4f}",
                        f"{residuals_df['Residuals'].max():.4f}",
                        f"{stats.shapiro(residuals_df['Residuals'])[1]:.6f}"
                    ]
                })
                
                st.dataframe(res_stats, use_container_width=True)
                
                # Interpretation of Shapiro-Wilk test
                shapiro_p = stats.shapiro(residuals_df['Residuals'])[1]
                if shapiro_p < 0.05:
                    st.warning("""
                    **Non-normal Residuals**: The Shapiro-Wilk test indicates that residuals are not normally distributed 
                    (p < 0.05). This may affect the validity of confidence intervals and hypothesis tests.
                    """)
                else:
                    st.success("""
                    **Normal Residuals**: The Shapiro-Wilk test suggests the residuals follow a normal distribution 
                    (p ≥ 0.05), which is good for the validity of the linear regression model.
                    """)
            
            with eval_tab3:
                st.subheader("Feature Importance")
                
                if coefficients:
                    # Create a dataframe from the coefficients
                    coef_items = list(coefficients.items())
                    # Move intercept to the end if it exists
                    if 'Intercept' in coefficients:
                        intercept_value = coefficients['Intercept']
                        coef_items = [(k, v) for k, v in coef_items if k != 'Intercept']
                        coef_items.append(('Intercept', intercept_value))
                    
                    coef_df = pd.DataFrame(coef_items, columns=['Feature', 'Coefficient'])
                    
                    # Handle feature scaling for interpretation
                    if 'preprocessor' in pipeline.named_steps and any(feat in feature_names for feat in numeric_columns):
                        st.info("""
                        **Note on Interpretation**: Since feature scaling was applied, coefficient magnitudes
                        are directly comparable to each other. Larger absolute values indicate stronger effects.
                        """)
                    else:
                        st.info("""
                        **Note on Interpretation**: Since feature scaling was not applied, 
                        coefficient magnitudes cannot be directly compared between features with different scales.
                        """)
                    
                    # Add a standard coefficient column (normalized coefficient)
                    coef_no_intercept = coef_df[coef_df['Feature'] != 'Intercept']
                    if not coef_no_intercept.empty:
                        max_abs_coef = max(abs(coef_no_intercept['Coefficient']))
                        coef_df['Standardized Coefficient'] = coef_df['Coefficient'].apply(
                            lambda x: x / max_abs_coef if coef_df.loc[coef_df['Coefficient'] == x, 'Feature'].iloc[0] != 'Intercept' else None
                        )
                    
                    # Sort by absolute coefficient value
                    coef_no_intercept = coef_df[coef_df['Feature'] != 'Intercept'].copy()
                    coef_no_intercept['Abs_Coefficient'] = coef_no_intercept['Coefficient'].abs()
                    coef_no_intercept = coef_no_intercept.sort_values('Abs_Coefficient', ascending=False)
                    
                    # Prepare for visualization - top N features
                    top_n = min(15, len(coef_no_intercept))  # Limit to 15 features for readability
                    top_features = coef_no_intercept.head(top_n)
                    
                    # Create horizontal bar chart
                    st.subheader(f"Top {top_n} Feature Coefficients by Magnitude")
                    
                    fig = px.bar(
                        top_features,
                        y='Feature',
                        x='Coefficient',
                        orientation='h',
                        title=f'Top {top_n} Features by Coefficient Magnitude',
                        color='Coefficient',
                        color_continuous_scale=px.colors.diverging.RdBu_r,
                        color_continuous_midpoint=0
                    )
                    
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Standardized coefficients visualization
                    if 'Standardized Coefficient' in top_features.columns:
                        st.subheader("Standardized Coefficients")
                        
                        fig = px.bar(
                            top_features,
                            y='Feature',
                            x='Standardized Coefficient',
                            orientation='h',
                            title='Standardized Coefficients (normalized to -1 to 1 range)',
                            color='Standardized Coefficient',
                            color_continuous_scale=px.colors.diverging.RdBu_r,
                            color_continuous_midpoint=0
                        )
                        
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display all coefficients in a table
                    st.subheader("All Coefficients")
                    st.dataframe(coef_df, use_container_width=True)
                    
                    # Create regression equation
                    st.subheader("Regression Equation")
                    
                    # Format the equation
                    intercept = coefficients.get('Intercept', 0)
                    equation = f"{target_col} = {intercept:.4f}"
                    
                    # Add top terms (limit to avoid overwhelming equation)
                    eq_terms = []
                    for feature, coef in coefficients.items():
                        if feature != 'Intercept':
                            if coef >= 0:
                                eq_terms.append(f"+ {coef:.4f} × {feature}")
                            else:
                                eq_terms.append(f"- {abs(coef):.4f} × {feature}")
                    
                    # If there are many terms, limit to top N by magnitude
                    if len(eq_terms) > 10:
                        # Get top 10 features by coefficient magnitude
                        top_features = coef_no_intercept.head(10)['Feature'].tolist()
                        filtered_eq_terms = []
                        for term in eq_terms:
                            # Extract feature name from term
                            feature = term.split('×')[1].strip()
                            if feature in top_features:
                                filtered_eq_terms.append(term)
                        
                        # Add terms to equation
                        equation += " " + " ".join(filtered_eq_terms)
                        equation += " + ..."  # Indicate there are more terms
                    else:
                        # Add all terms to equation
                        equation += " " + " ".join(eq_terms)
                    
                    st.code(equation)
                    
                    # Feature correlation with target
                    st.subheader("Feature Correlation with Target")
                    
                    # Calculate correlations between numeric features and target
                    corr_data = []
                    for feature in feature_names:
                        if feature in numeric_columns:
                            corr = df[[feature, target_col]].corr().iloc[0, 1]
                            corr_data.append({
                                'Feature': feature,
                                'Correlation': corr,
                                'Abs_Correlation': abs(corr)
                            })
                    
                    if corr_data:
                        corr_df = pd.DataFrame(corr_data)
                        corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
                        
                        fig = px.bar(
                            corr_df,
                            y='Feature',
                            x='Correlation',
                            orientation='h',
                            title='Correlation with Target Variable',
                            color='Correlation',
                            color_continuous_scale=px.colors.diverging.RdBu_r,
                            color_continuous_midpoint=0
                        )
                        
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add explanation
                        st.markdown("""
                        **Interpreting Correlations vs. Coefficients:**
                        
                        - **Correlation** shows the direct relationship between each feature and the target
                        - **Coefficients** show the relationship accounting for all other features in the model
                        - Differences between them may indicate complex interactions or multicollinearity
                        """)
                    else:
                        st.info("No numeric features available for correlation analysis.")
                else:
                    st.warning("Could not extract model coefficients for analysis.")
                
                # Model summary
                st.subheader("Model Summary")
                
                # Create a summary table
                summary_data = {
                    'Metric': ['Model Type', 'Features', 'R² Score', 'RMSE', 'MAE', 'Training Size', 'Test Size'],
                    'Value': [
                        'Linear Regression',
                        f"{len(feature_names)} features",
                        f"{metrics['R² Score']:.4f}",
                        f"{metrics['Root Mean Squared Error']:.4f}",
                        f"{metrics['Mean Absolute Error']:.4f}",
                        f"{len(X_train)} samples",
                        f"{len(X_test)} samples"
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Final conclusions
                st.subheader("Model Assessment")
                
                # Determine if R² is good based on common thresholds
                r2 = metrics['R² Score']
                
                if r2 < 0:
                    assessment = "Poor (worse than mean predictor)"
                    color = "red"
                elif r2 < 0.2:
                    assessment = "Very weak"
                    color = "red"
                elif r2 < 0.4:
                    assessment = "Weak"
                    color = "orange"
                elif r2 < 0.6:
                    assessment = "Moderate"
                    color = "blue"
                elif r2 < 0.8:
                    assessment = "Strong"
                    color = "green"
                else:
                    assessment = "Very strong"
                    color = "green"
                
                st.markdown(f"**Model Fit Quality:** <span style='color:{color}'>{assessment}</span> (R² = {r2:.4f})", unsafe_allow_html=True)
                
                # Recommendations based on model performance
                st.subheader("Recommendations")
                
                recommendations = []
                
                if r2 < 0.4:
                    recommendations.append("Consider adding more relevant features to improve predictive power")
                    recommendations.append("Try non-linear models that might better capture complex relationships")
                    recommendations.append("Check for outliers that might be affecting model performance")
                
                if abs(np.mean(residuals_df['Residuals'])) > 0.1 * np.std(residuals_df['Residuals']):
                    recommendations.append("The model shows some bias (non-zero mean residuals). Consider feature engineering or transformation")
                
                shapiro_p = stats.shapiro(residuals_df['Residuals'])[1]
                if shapiro_p < 0.05:
                    recommendations.append("Residuals are not normally distributed. Consider transforming the target variable (e.g., log transform)")
                
                # Check for patterns in residuals (simplified)
                residual_corr = np.corrcoef(residuals_df['Predicted'], residuals_df['Residuals'])[0, 1]
                if abs(residual_corr) > 0.1:
                    recommendations.append(f"Residuals show correlation with predicted values ({residual_corr:.4f}). The relationship may be non-linear")
                
                if r2_train > r2 * 1.2:
                    recommendations.append("Model shows signs of overfitting. Consider regularization techniques")
                
                if not recommendations:
                    recommendations.append("The model appears well-fitted to the data")
                    recommendations.append("Consider using this model for predictions if the performance meets requirements")
                
                for rec in recommendations:
                    st.markdown(f"- {rec}")
                
                # Next steps
                st.markdown("""
                **Next Steps:**
                - Use the 'Make Predictions' tab to apply this model to new data
                - Experiment with feature selection and transformation to improve model performance
                - Consider ensemble methods or more complex models if needed
                """)

    with tab4:
        st.header("Make Predictions")
        
        # Check if a model has been trained
        if st.session_state.get('lr_trained_model') is None:
            st.warning("⚠️ You need to train a model first. Go to the 'Train Model' tab to train a Linear Regression model.")
        else:
            # Get model and data from session state
            pipeline = st.session_state.lr_pipeline
            feature_names = st.session_state.lr_feature_names
            target_col = st.session_state.lr_target
            
            # Create subtabs for different prediction types
            pred_tab1, pred_tab2, pred_tab3 = st.tabs(["Single Prediction", "Batch Predictions", "Upload File"])
            
            with pred_tab1:
                st.subheader("Make a Single Prediction")
                st.markdown("Enter values for each feature to get a prediction.")
                
                # Create a form for input values
                with st.form("single_prediction_form"):
                    # Create input fields for each feature
                    input_values = {}
                    
                    # Group features in columns (3 per row)
                    num_features = len(feature_names)
                    num_rows = (num_features + 2) // 3  # Ceiling division
                    
                    for row in range(num_rows):
                        cols = st.columns(3)
                        for col_idx in range(3):
                            feature_idx = row * 3 + col_idx
                            if feature_idx < num_features:
                                feature = feature_names[feature_idx]
                                
                                # Determine input type based on original dataframe
                                if df[feature].dtype in ['int64', 'float64']:
                                    # Numeric input
                                    min_val = float(df[feature].min())
                                    max_val = float(df[feature].max())
                                    mean_val = float(df[feature].mean())
                                    
                                    # Use slider for numeric input
                                    input_values[feature] = cols[col_idx].slider(
                                        f"{feature}",
                                        min_value=min_val,
                                        max_value=max_val,
                                        value=mean_val,
                                        format="%.2f" if df[feature].dtype == 'float64' else "%d"
                                    )
                                else:
                                    # Categorical input (dropdown)
                                    unique_values = df[feature].dropna().unique().tolist()
                                    input_values[feature] = cols[col_idx].selectbox(
                                        f"{feature}",
                                        options=unique_values,
                                        index=0
                                    )
                    
                    # Submit button
                    predict_button = st.form_submit_button("Make Prediction")
                
                # Make prediction when button is clicked
                if predict_button:
                    # Convert input to DataFrame
                    input_df = pd.DataFrame([input_values])
                    
                    # Make prediction
                    prediction = pipeline.predict(input_df)[0]
                    
                    # Display prediction
                    st.success(f"### Predicted {target_col}: {prediction:.4f}")
                    
                    # Calculate confidence interval (approximation)
                    # Note: This is a simple approximation based on RMSE
                    if 'Root Mean Squared Error' in st.session_state.lr_model_metrics:
                        rmse = st.session_state.lr_model_metrics['Root Mean Squared Error']
                        lower_bound = prediction - 1.96 * rmse
                        upper_bound = prediction + 1.96 * rmse
                        
                        st.info(f"""
                        **Approximate 95% Prediction Interval:** 
                        {lower_bound:.4f} to {upper_bound:.4f}
                        
                        This interval represents the range where we expect the actual value to fall 
                        with 95% confidence, based on the model's error rate.
                        """)
                    
                    # Feature impact analysis
                    st.subheader("Feature Impact Analysis")
                    
                    # Extract coefficients if available
                    if st.session_state.lr_coefficients:
                        coefficients = st.session_state.lr_coefficients
                        
                        # Create a feature impact visualization
                        try:
                            # Get the processed features
                            preprocessor = pipeline.named_steps['preprocessor']
                            X_processed = preprocessor.transform(input_df)
                            
                            # Get the feature names after preprocessing
                            feature_names_out = []
                            if hasattr(preprocessor, 'transformers_'):
                                for name, transformer, features in preprocessor.transformers_:
                                    if name == 'cat' and hasattr(transformer, 'named_steps') and 'onehot' in transformer.named_steps:
                                        onehot = transformer.named_steps['onehot']
                                        if hasattr(onehot, 'get_feature_names_out'):
                                            cat_features = onehot.get_feature_names_out(features)
                                            feature_names_out.extend(cat_features)
                                        else:
                                            feature_names_out.extend([f"{f}_encoded" for f in features])
                                    else:
                                        feature_names_out.extend(features)
                            
                            # Calculate feature contributions
                            impact_data = []
                            
                            # If we have transformed feature names
                            if feature_names_out and len(coefficients) > 1:  # More than just intercept
                                # Filter out intercept
                                feature_coeffs = {k: v for k, v in coefficients.items() if k != 'Intercept'}
                                
                                # Calculate impact values
                                for feature, coef in feature_coeffs.items():
                                    # Find the feature index
                                    try:
                                        # Try exact match
                                        feature_idx = feature_names_out.index(feature)
                                        feature_value = X_processed[0, feature_idx]
                                        impact = coef * feature_value
                                        
                                        impact_data.append({
                                            'Feature': feature,
                                            'Impact': impact,
                                            'Abs_Impact': abs(impact)
                                        })
                                    except ValueError:
                                        # If not found, this might be a transformed feature we can't directly map
                                        pass
                                
                                if impact_data:
                                    # Create impact dataframe
                                    impact_df = pd.DataFrame(impact_data)
                                    impact_df = impact_df.sort_values('Abs_Impact', ascending=False)
                                    
                                    # Add intercept as base value
                                    intercept = coefficients.get('Intercept', 0)
                                    
                                    # Create waterfall chart
                                    st.markdown("### Feature Impact Waterfall Chart")
                                    st.markdown("This chart shows how each feature contributes to the final prediction.")
                                    
                                    # Show top impacts
                                    top_n = min(10, len(impact_df))  # Limit to top 10 for readability
                                    
                                    # Calculate cumulative impact for waterfall
                                    impact_df = impact_df.head(top_n)
                                    
                                    # Create the base for waterfall chart
                                    measure = ['absolute'] + ['relative'] * len(impact_df) + ['total']
                                    
                                    # Feature names for x-axis
                                    x_data = ['Base'] + impact_df['Feature'].tolist() + ['Prediction']
                                    
                                    # Values for y-axis
                                    y_data = [intercept]
                                    for impact in impact_df['Impact']:
                                        y_data.append(impact)
                                    y_data.append(0)  # Placeholder for the total
                                    
                                    # Create waterfall chart using plotly
                                    fig = go.Figure(go.Waterfall(
                                        name="Feature Impact",
                                        orientation="v",
                                        measure=measure,
                                        x=x_data,
                                        y=y_data,
                                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                                        decreasing={"marker": {"color": "Crimson"}},
                                        increasing={"marker": {"color": "ForestGreen"}},
                                        totals={"marker": {"color": "DeepSkyBlue"}}
                                    ))
                                    
                                    fig.update_layout(
                                        title="Feature Impact Waterfall Chart",
                                        showlegend=False,
                                        xaxis_title="Feature",
                                        yaxis_title="Impact on Prediction",
                                        waterfallgap=0.2,
                                        xaxis={'type': 'category'}
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Explanation of the waterfall chart
                                    st.markdown("""
                                    **How to interpret this chart:**
                                    
                                    - **Base**: The starting value (intercept) when all features are zero
                                    - **Green bars**: Features that increase the predicted value
                                    - **Red bars**: Features that decrease the predicted value
                                    - **Blue bar**: The final prediction after all feature impacts
                                    
                                    The height of each bar shows how much that feature contributes to the prediction.
                                    """)
                                else:
                                    st.info("Could not calculate detailed feature impacts for this prediction.")
                            else:
                                st.info("Feature impact analysis not available for this model.")
                        
                        except Exception as e:
                            st.error(f"Error calculating feature impacts: {str(e)}")
                            st.info("Feature impact analysis could not be completed.")
                    else:
                        st.info("No coefficient information available for impact analysis.")
            
            with pred_tab2:
                st.subheader("Make Batch Predictions")
                st.markdown("Use the current dataset to make predictions for multiple samples.")
                
                # Select which part of the dataset to use
                data_option = st.radio(
                    "Select data for prediction:",
                    options=["Use test set (if available)", "Use entire dataset", "Use random samples"],
                    horizontal=True
                )
                
                # Get data based on selection
                if data_option == "Use test set (if available)":
                    if st.session_state.get('lr_X_test') is not None:
                        X_pred = st.session_state.lr_X_test
                        y_true = st.session_state.lr_y_test
                        st.info(f"Using {len(X_pred)} samples from the test set")
                    else:
                        X_pred = df[feature_names]
                        y_true = None
                        st.warning("Test set not available. Using entire dataset.")
                elif data_option == "Use random samples":
                    # Select number of random samples
                    n_samples = st.slider(
                        "Number of random samples:",
                        min_value=1,
                        max_value=min(100, len(df)),
                        value=min(10, len(df))
                    )
                    
                    # Get random samples
                    random_indices = np.random.choice(len(df), size=n_samples, replace=False)
                    X_pred = df.iloc[random_indices][feature_names]
                    y_true = df.iloc[random_indices][target_col] if target_col in df.columns else None
                    st.info(f"Using {len(X_pred)} random samples from the dataset")
                else:  # Use entire dataset
                    X_pred = df[feature_names]
                    y_true = df[target_col] if target_col in df.columns else None
                    st.info(f"Using all {len(X_pred)} samples from the dataset")
                
                # Number of samples to display
                display_limit = st.slider(
                    "Number of results to display:",
                    min_value=1,
                    max_value=min(100, len(X_pred)),
                    value=min(10, len(X_pred))
                )
                
                # Make predictions button
                if st.button("Generate Predictions"):
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        status_text.text("Making predictions...")
                        
                        # Make predictions
                        y_pred = pipeline.predict(X_pred)
                        
                        progress_bar.progress(50)
                        status_text.text("Processing results...")
                        
                        # Create results DataFrame
                        results_df = X_pred.copy()
                        results_df['Predicted'] = y_pred
                        
                        # If actual values are available
                        if y_true is not None:
                            results_df['Actual'] = y_true
                            results_df['Error'] = y_true - y_pred
                            results_df['Absolute Error'] = abs(results_df['Error'])
                            results_df['Percentage Error'] = abs(results_df['Error'] / y_true) * 100
                            
                            # Calculate metrics
                            batch_mse = mean_squared_error(y_true, y_pred)
                            batch_r2 = r2_score(y_true, y_pred)
                            batch_mae = mean_absolute_error(y_true, y_pred)
                            
                            # Create metrics display
                            metrics_cols = st.columns(3)
                            with metrics_cols[0]:
                                st.metric("R² Score", f"{batch_r2:.4f}")
                            with metrics_cols[1]:
                                st.metric("RMSE", f"{np.sqrt(batch_mse):.4f}")
                            with metrics_cols[2]:
                                st.metric("MAE", f"{batch_mae:.4f}")
                        
                        progress_bar.progress(100)
                        status_text.text("Predictions complete!")
                        
                        # Display results
                        st.subheader("Prediction Results")
                        st.dataframe(results_df.head(display_limit), use_container_width=True)
                        
                        # Download results
                        st.download_button(
                            "Download All Prediction Results",
                            results_df.to_csv(index=False).encode('utf-8'),
                            "linear_regression_predictions.csv",
                            "text/csv",
                            key='download-batch-predictions'
                        )
                        
                        # Visualization of predictions
                        if y_true is not None:
                            st.subheader("Actual vs. Predicted Values")
                            
                            fig = px.scatter(
                                results_df,
                                x='Actual',
                                y='Predicted',
                                title='Actual vs. Predicted Values',
                                color='Absolute Error',
                                color_continuous_scale='Viridis',
                                labels={
                                    'Actual': f'Actual {target_col}',
                                    'Predicted': f'Predicted {target_col}',
                                    'Absolute Error': 'Absolute Error'
                                }
                            )
                            
                            # Add perfect prediction line
                            min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
                            max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
                            fig.add_trace(
                                go.Scatter(
                                    x=[min_val, max_val],
                                    y=[min_val, max_val],
                                    mode='lines',
                                    name='Perfect Prediction',
                                    line=dict(color='red', dash='dash')
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Error distribution
                            st.subheader("Error Distribution")
                            
                            fig = px.histogram(
                                results_df,
                                x='Error',
                                title='Distribution of Prediction Errors',
                                marginal='box'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        else:
                            # Just show distribution of predictions
                            st.subheader("Prediction Distribution")
                            
                            fig = px.histogram(
                                results_df,
                                x='Predicted',
                                title=f'Distribution of Predicted {target_col}',
                                marginal='box'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"An error occurred during prediction: {str(e)}")
            
            with pred_tab3:
                st.subheader("Upload File for Predictions")
                st.markdown("""
                Upload a CSV or Excel file with the same feature columns as your training data.
                The model will generate predictions for each row.
                """)
                
                # File upload
                uploaded_file = st.file_uploader("Upload your file", type=['csv', 'xlsx', 'xls'])
                
                if uploaded_file is not None:
                    try:
                        # Read file
                        if uploaded_file.name.endswith('.csv'):
                            predict_df = pd.read_csv(uploaded_file)
                        else:
                            predict_df = pd.read_excel(uploaded_file)
                        
                        # Show file preview
                        st.subheader("File Preview")
                        st.dataframe(predict_df.head(5), use_container_width=True)
                        
                        # Check if required features are present
                        missing_features = [f for f in feature_names if f not in predict_df.columns]
                        
                        if missing_features:
                            st.error(f"Missing required features: {', '.join(missing_features)}")
                            st.info(f"Your file must contain all the features used during training: {', '.join(feature_names)}")
                        else:
                            # Extract features
                            X_upload = predict_df[feature_names]
                            
                            # Number of samples to display
                            display_limit = st.slider(
                                "Number of results to display:",
                                min_value=1,
                                max_value=min(100, len(X_upload)),
                                value=min(10, len(X_upload))
                            )
                            
                            # Check if target column exists in uploaded file
                            has_target = target_col in predict_df.columns
                            
                            # Make predictions button
                            if st.button("Generate Predictions from File"):
                                # Show progress
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                try:
                                    status_text.text("Making predictions...")
                                    
                                    # Make predictions
                                    y_pred = pipeline.predict(X_upload)
                                    
                                    progress_bar.progress(50)
                                    status_text.text("Processing results...")
                                    
                                    # Create results DataFrame (include all original columns)
                                    results_df = predict_df.copy()
                                    results_df[f'Predicted_{target_col}'] = y_pred
                                    
                                    # If actual target is available, calculate errors
                                    if has_target:
                                        results_df['Error'] = results_df[target_col] - results_df[f'Predicted_{target_col}']
                                        results_df['Absolute Error'] = abs(results_df['Error'])
                                        
                                        # Calculate metrics
                                        y_true = predict_df[target_col]
                                        upload_mse = mean_squared_error(y_true, y_pred)
                                        upload_r2 = r2_score(y_true, y_pred)
                                        upload_mae = mean_absolute_error(y_true, y_pred)
                                        
                                        # Display metrics
                                        metrics_cols = st.columns(3)
                                        with metrics_cols[0]:
                                            st.metric("R² Score", f"{upload_r2:.4f}")
                                        with metrics_cols[1]:
                                            st.metric("RMSE", f"{np.sqrt(upload_mse):.4f}")
                                        with metrics_cols[2]:
                                            st.metric("MAE", f"{upload_mae:.4f}")
                                    
                                    progress_bar.progress(100)
                                    status_text.text("Predictions complete!")
                                    
                                    # Display results
                                    st.subheader("Prediction Results")
                                    st.dataframe(results_df.head(display_limit), use_container_width=True)
                                    
                                    # Download results
                                    st.download_button(
                                        "Download All Prediction Results",
                                        results_df.to_csv(index=False).encode('utf-8'),
                                        "linear_regression_file_predictions.csv",
                                        "text/csv",
                                        key='download-file-predictions'
                                    )
                                    
                                    # Visualizations
                                    if has_target:
                                        # Actual vs Predicted
                                        st.subheader("Actual vs. Predicted Values")
                                        
                                        fig = px.scatter(
                                            results_df,
                                            x=target_col,
                                            y=f'Predicted_{target_col}',
                                            title='Actual vs. Predicted Values',
                                            labels={
                                                target_col: f'Actual {target_col}',
                                                f'Predicted_{target_col}': f'Predicted {target_col}'
                                            }
                                        )
                                        
                                        # Add perfect prediction line
                                        min_val = min(results_df[target_col].min(), results_df[f'Predicted_{target_col}'].min())
                                        max_val = max(results_df[target_col].max(), results_df[f'Predicted_{target_col}'].max())
                                        fig.add_trace(
                                            go.Scatter(
                                                x=[min_val, max_val],
                                                y=[min_val, max_val],
                                                mode='lines',
                                                name='Perfect Prediction',
                                                line=dict(color='red', dash='dash')
                                            )
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        # Just prediction distribution
                                        st.subheader("Prediction Distribution")
                                        
                                        fig = px.histogram(
                                            results_df,
                                            x=f'Predicted_{target_col}',
                                            title=f'Distribution of Predicted {target_col}',
                                            marginal='box'
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                except Exception as e:
                                    st.error(f"An error occurred during prediction: {str(e)}")
                    
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
                
                # Instructions
                with st.expander("File Format Instructions"):
                    st.markdown(f"""
                    ### Required File Format
                    
                    Your file should:
                    
                    1. Be in CSV or Excel format
                    2. Contain columns for all features used in training
                    3. Have the same column names as the training data
                    
                    The required feature columns are:
                    ```
                    {', '.join(feature_names)}
                    ```
                    
                    Optionally, include a "{target_col}" column to compare predictions with actual values.
                    """)
            
            # Add model integration information section
            with st.expander("Model Integration Information"):
                st.markdown("""
                ### How to Use This Model in Your Applications
                
                To integrate this trained model into your own applications:
                
                1. Train the model and download it from the "Train Model" tab
                2. Load the model in your Python application using:
                   ```python
                   import pickle
                   
                   # Load the model
                   with open('linear_regression_model.pkl', 'rb') as f:
                       model = pickle.load(f)
                   
                   # Make predictions
                   predictions = model.predict(X_new)
                   ```
                
                3. Ensure your input data has the same format and features as the training data
                
                #### API Integration Example
                
                ```python
                from flask import Flask, request, jsonify
                import pandas as pd
                import pickle
                
                app = Flask(__name__)
                
                # Load the model
                with open('linear_regression_model.pkl', 'rb') as f:
                    model = pickle.load(f)
                
                @app.route('/predict', methods=['POST'])
                def predict():
                    # Get JSON data from request
                    data = request.get_json()
                    
                    # Convert to DataFrame
                    input_df = pd.DataFrame(data, index=[0])
                    
                    # Make prediction
                    prediction = model.predict(input_df)[0]
                    
                    # Return result
                    return jsonify({
                        'prediction': float(prediction)
                    })
                
                if __name__ == '__main__':
                    app.run(debug=True)
                ```
                """)

else:
    # Show warning if no data is loaded
    show_file_required_warning()

# Footer
create_footer() 
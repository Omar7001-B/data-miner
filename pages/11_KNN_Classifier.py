import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
import warnings
import io
import time
import pickle
import base64

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
st.set_page_config(page_title="KNN Classifier", page_icon="🧠", layout="wide")
load_css()

# Main title
st.title("K-Nearest Neighbors Classifier 🧠")
st.markdown("""
Train a K-Nearest Neighbors (KNN) model to classify data. This page helps you set up, train, 
evaluate, and visualize a KNN model for your classification problems.
""")

# Check if data is loaded
df = st.session_state.get('df', None)
if df is not None:
    # Display dataset info
    display_dataset_info()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Train Model", "Evaluate Model", "Make Predictions"])
    
    with tab1:
        st.header("KNN Overview")
        
        st.markdown("""
        ### What is K-Nearest Neighbors (KNN)?
        
        K-Nearest Neighbors (KNN) is a simple, yet powerful instance-based classification and regression algorithm. 
        It classifies data points based on the most common class among their k-nearest neighbors in the feature space.
        
        ### When to Use KNN
        
        KNN works well when:
        - The decision boundary is irregular or complex
        - You have relatively small datasets (KNN can be computationally expensive for large datasets)
        - Features are meaningful in a distance-based context
        - The data has low dimensionality (or has been reduced)
        
        ### Advantages of KNN
        
        - **Simplicity**: Easy to understand and implement
        - **No training phase**: It's a lazy learning algorithm that stores the training data
        - **Adaptability**: Naturally handles multi-class problems
        - **Non-parametric**: Makes no assumptions about the underlying data distribution
        
        ### Limitations
        
        - Computationally expensive for large datasets
        - Feature scaling is mandatory (distance-based algorithm)
        - Sensitive to irrelevant features and the curse of dimensionality
        - Requires feature selection or dimensionality reduction for high-dimensional data
        - Storage requirements can be large
        """)
        
        # Show dataset preview
        st.subheader("Dataset Preview")
        st.dataframe(df.head(5), use_container_width=True)
        
        # Check for categorical target variables
        categorical_columns = get_categorical_columns(df)
        if categorical_columns:
            st.subheader("Potential Target Variables")
            st.markdown("These categorical columns could be suitable targets for classification:")
            
            # Show potential target variables with class distribution
            for col in categorical_columns[:5]:  # Limit to first 5 to avoid cluttering the UI
                try:
                    # Calculate value counts and percentages
                    val_counts = df[col].value_counts()
                    val_percentages = df[col].value_counts(normalize=True) * 100
                    
                    # If too many categories, only show top 5
                    if len(val_counts) > 5:
                        st.markdown(f"**{col}** (showing top 5 of {len(val_counts)} categories)")
                        counts_df = pd.DataFrame({
                            'Count': val_counts[:5],
                            'Percentage': val_percentages[:5].round(2)
                        })
                        st.dataframe(counts_df, use_container_width=True)
                    else:
                        st.markdown(f"**{col}**")
                        counts_df = pd.DataFrame({
                            'Count': val_counts,
                            'Percentage': val_percentages.round(2)
                        })
                        st.dataframe(counts_df, use_container_width=True)
                    
                    # Create bar chart for class distribution
                    fig = px.bar(
                        x=val_counts.index[:10],  # Limit to first 10 categories for visualization
                        y=val_counts.values[:10],
                        labels={'x': 'Category', 'y': 'Count'},
                        title=f"Class Distribution for {col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not analyze column {col}: {str(e)}")
        else:
            st.info("No categorical columns detected in this dataset. To use a KNN classifier, you need a categorical target variable.")
            
    with tab2:
        st.header("Train KNN Model")
        
        # Initialize session state variables for model training
        if 'knn_target' not in st.session_state:
            st.session_state.knn_target = None
        if 'knn_features' not in st.session_state:
            st.session_state.knn_features = []
        if 'knn_trained_model' not in st.session_state:
            st.session_state.knn_trained_model = None
        if 'knn_model_metrics' not in st.session_state:
            st.session_state.knn_model_metrics = None
        if 'knn_pipeline' not in st.session_state:
            st.session_state.knn_pipeline = None
        if 'knn_feature_names' not in st.session_state:
            st.session_state.knn_feature_names = None
        if 'knn_classes' not in st.session_state:
            st.session_state.knn_classes = None
        if 'knn_X_train' not in st.session_state:
            st.session_state.knn_X_train = None
        if 'knn_X_test' not in st.session_state:
            st.session_state.knn_X_test = None
        if 'knn_y_train' not in st.session_state:
            st.session_state.knn_y_train = None
        if 'knn_y_test' not in st.session_state:
            st.session_state.knn_y_test = None
        if 'knn_best_k' not in st.session_state:
            st.session_state.knn_best_k = None
        
        # Step 1: Select target variable
        st.subheader("Step 1: Select Target Variable")
        
        # Get categorical columns with 2-10 unique values (ideal for classification)
        suitable_targets = []
        for col in df.columns:
            if col in categorical_columns or df[col].nunique() <= 10:
                n_classes = df[col].nunique()
                if 2 <= n_classes <= 10:  # Binary or multiclass with reasonable number of classes
                    suitable_targets.append(col)
        
        if not suitable_targets:
            st.warning("No suitable categorical target variables found. Ideal target variables should have 2-10 unique classes.")
            st.info("You can convert a numeric column to categorical in the 'Categorical Conversion' page.")
            suitable_targets = categorical_columns  # Use all categorical columns as fallback
        
        target_col = st.selectbox(
            "Select target variable (what you want to predict):",
            options=suitable_targets,
            index=0 if suitable_targets else None,
            help="The column containing the classes you want to predict"
        )
        
        if target_col:
            st.session_state.knn_target = target_col
            
            # Show distribution of target variable
            target_counts = df[target_col].value_counts()
            
            # Create target distribution chart
            fig = px.pie(
                values=target_counts.values,
                names=target_counts.index,
                title=f"Distribution of {target_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Step 2: Select features
            st.subheader("Step 2: Select Features")
            
            st.info("""
            **Note:** KNN is sensitive to the curse of dimensionality. For best results:
            - Select only the most relevant features
            - Use fewer features for better performance
            - Consider using continuous (numerical) features
            """)
            
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
                    default=st.session_state.knn_features if st.session_state.knn_features else []
                )
            
            # Update session state
            st.session_state.knn_features = selected_features
            
            if not selected_features:
                st.warning("Please select at least one feature for training.")
            
            # Step 3: Configure model parameters
            if selected_features:
                st.subheader("Step 3: Configure Model Parameters")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # KNN-specific parameters
                    n_neighbors = st.slider(
                        "Number of neighbors (k):",
                        min_value=1,
                        max_value=20,
                        value=5,
                        help="Number of neighbors to consider. Higher values reduce noise but might smooth out decision boundaries."
                    )
                    
                    weights = st.selectbox(
                        "Weighting:",
                        options=["uniform", "distance"],
                        index=0,
                        help="'uniform': all neighbors weighted equally, 'distance': closer neighbors have greater influence"
                    )
                    
                    distance_metric = st.selectbox(
                        "Distance metric:",
                        options=["euclidean", "manhattan", "minkowski"],
                        index=0,
                        help="Method to calculate distance between points. Euclidean is the straight-line distance."
                    )
                    
                    if distance_metric == "minkowski":
                        p_value = st.slider(
                            "p value for Minkowski:",
                            min_value=1,
                            max_value=10,
                            value=2,
                            help="Power parameter for Minkowski metric. p=1 is Manhattan, p=2 is Euclidean."
                        )
                    else:
                        p_value = 2  # Default for Euclidean
                
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
                        help="Standardize features for better KNN performance (recommended)"
                    )
                    
                    auto_k_selection = st.checkbox(
                        "Find optimal k value",
                        value=False,
                        help="Automatically find the best k value using cross-validation"
                    )
                    
                    if auto_k_selection:
                        max_k_to_test = st.slider(
                            "Maximum k to test:",
                            min_value=1,
                            max_value=30,
                            value=15,
                            help="The maximum number of neighbors to test"
                        )
                
                # Step 4: Train the model
                st.subheader("Step 4: Train the Model")
                
                train_button = st.button("Train KNN Model")
                
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
                                X = X.dropna(axis=0)
                                y = y.loc[X.index]
                            
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
                            
                            # Create label encoder for target
                            le = LabelEncoder()
                            y_encoded = le.fit_transform(y)
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y_encoded, 
                                test_size=test_size, 
                                random_state=int(random_state),
                                stratify=y_encoded  # Ensure balanced classes in train and test sets
                            )
                            
                            progress_bar.progress(30)
                            
                            # Automated k selection if requested
                            if auto_k_selection:
                                status_text.text("Finding optimal k value...")
                                
                                # Preprocess the data
                                X_train_processed = preprocessor.fit_transform(X_train)
                                X_test_processed = preprocessor.transform(X_test)
                                
                                # Test different k values
                                k_values = range(1, max_k_to_test + 1)
                                accuracy_scores = []
                                
                                # Progress counter for k selection
                                k_progress_step = 40 / len(k_values)
                                k_progress = 30
                                
                                for k in k_values:
                                    # Create and train KNN model
                                    knn = KNeighborsClassifier(
                                        n_neighbors=k,
                                        weights=weights,
                                        metric=distance_metric,
                                        p=p_value if distance_metric == 'minkowski' else 2
                                    )
                                    
                                    knn.fit(X_train_processed, y_train)
                                    
                                    # Calculate accuracy
                                    y_pred = knn.predict(X_test_processed)
                                    accuracy = accuracy_score(y_test, y_pred)
                                    accuracy_scores.append(accuracy)
                                    
                                    # Update progress
                                    k_progress += k_progress_step
                                    progress_bar.progress(int(k_progress))
                                
                                # Find best k
                                best_k_idx = np.argmax(accuracy_scores)
                                best_k = k_values[best_k_idx]
                                
                                # Show k selection results
                                st.session_state.knn_best_k = best_k
                                
                                # Create line chart of k values vs accuracy
                                k_results_fig = px.line(
                                    x=list(k_values),
                                    y=accuracy_scores,
                                    markers=True,
                                    labels={'x': 'k value', 'y': 'Accuracy'},
                                    title="Accuracy vs k Value"
                                )
                                
                                # Add vertical line at best k
                                k_results_fig.add_vline(
                                    x=best_k, 
                                    line_dash="dash", 
                                    line_color="green",
                                    annotation_text=f"Best k = {best_k}",
                                    annotation_position="top right"
                                )
                                
                                # Update n_neighbors with best k
                                n_neighbors = best_k
                                
                                status_text.text(f"Best k value found: {best_k}")
                            
                            progress_bar.progress(70)
                            status_text.text("Training model...")
                            
                            # Create model with selected parameters
                            model = KNeighborsClassifier(
                                n_neighbors=n_neighbors,
                                weights=weights,
                                metric=distance_metric,
                                p=p_value if distance_metric == 'minkowski' else 2
                            )
                            
                            # Create full pipeline
                            pipeline = Pipeline(steps=[
                                ('preprocessor', preprocessor),
                                ('classifier', model)
                            ])
                            
                            # Train model
                            start_time = time.time()
                            pipeline.fit(X_train, y_train)
                            training_time = time.time() - start_time
                            
                            progress_bar.progress(80)
                            status_text.text("Evaluating model...")
                            
                            # Make predictions
                            y_pred = pipeline.predict(X_test)
                            
                            # For multiclass problems, compute class-specific metrics
                            classes = le.classes_
                            is_multiclass = len(classes) > 2
                            
                            # Calculate metrics
                            metrics = {
                                "Accuracy": accuracy_score(y_test, y_pred),
                                "Training Time": training_time,
                                "Test Set Size": len(X_test),
                                "k Value": n_neighbors
                            }
                            
                            # Classification metrics
                            if not is_multiclass:
                                metrics.update({
                                    "Precision": precision_score(y_test, y_pred, average='binary'),
                                    "Recall": recall_score(y_test, y_pred, average='binary'),
                                    "F1 Score": f1_score(y_test, y_pred, average='binary')
                                })
                            else:
                                # Multiclass metrics
                                metrics.update({
                                    "Precision (macro)": precision_score(y_test, y_pred, average='macro'),
                                    "Recall (macro)": recall_score(y_test, y_pred, average='macro'),
                                    "F1 Score (macro)": f1_score(y_test, y_pred, average='macro')
                                })
                            
                            progress_bar.progress(90)
                            
                            # Save to session state
                            st.session_state.knn_trained_model = model
                            st.session_state.knn_model_metrics = metrics
                            st.session_state.knn_pipeline = pipeline
                            st.session_state.knn_feature_names = selected_features
                            st.session_state.knn_classes = classes
                            st.session_state.knn_X_train = X_train
                            st.session_state.knn_X_test = X_test
                            st.session_state.knn_y_train = y_train
                            st.session_state.knn_y_test = y_test
                            
                            progress_bar.progress(100)
                            status_text.text("Training complete!")
                            
                            # Display results
                            st.success("✅ Model training complete!")
                            
                            # Display metrics in a nice format
                            st.subheader("Model Performance Metrics")
                            
                            # Create metrics in a grid
                            metric_cols1 = st.columns(4)
                            
                            with metric_cols1[0]:
                                st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                            
                            with metric_cols1[1]:
                                if not is_multiclass:
                                    st.metric("Precision", f"{metrics['Precision']:.4f}")
                                else:
                                    st.metric("Precision (macro)", f"{metrics['Precision (macro)']:.4f}")
                            
                            with metric_cols1[2]:
                                if not is_multiclass:
                                    st.metric("Recall", f"{metrics['Recall']:.4f}")
                                else:
                                    st.metric("Recall (macro)", f"{metrics['Recall (macro)']:.4f}")
                            
                            with metric_cols1[3]:
                                if not is_multiclass:
                                    st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                                else:
                                    st.metric("F1 Score (macro)", f"{metrics['F1 Score (macro)']:.4f}")
                            
                            # KNN-specific metrics
                            metric_cols2 = st.columns(3)
                            
                            with metric_cols2[0]:
                                st.metric("k Value", metrics["k Value"])
                            
                            with metric_cols2[1]:
                                st.metric("Test Set Size", f"{metrics['Test Set Size']} samples")
                            
                            with metric_cols2[2]:
                                st.metric("Training Time", f"{metrics['Training Time']:.2f} sec")
                            
                            # Display confusion matrix
                            st.subheader("Confusion Matrix")
                            cm = confusion_matrix(y_test, y_pred)
                            
                            # Create heatmap of confusion matrix
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(
                                cm, 
                                annot=True, 
                                fmt='d', 
                                cmap='Blues',
                                xticklabels=classes,
                                yticklabels=classes
                            )
                            plt.ylabel('True Label')
                            plt.xlabel('Predicted Label')
                            plt.title('Confusion Matrix')
                            st.pyplot(fig)
                            
                            # Show k selection results if automated selection was used
                            if auto_k_selection and 'k_results_fig' in locals():
                                st.subheader("K Value Selection Results")
                                st.plotly_chart(k_results_fig, use_container_width=True)
                                st.info(f"The optimal k value was determined to be {best_k}")
                            
                            # Show classification report
                            st.subheader("Classification Report")
                            report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df)
                            
                            # Prompt to go to evaluation tab
                            st.info("👉 Go to the 'Evaluate Model' tab to explore model details and visualizations.")
                            
                            # Save model button
                            st.subheader("Save Trained Model")
                            
                            # Save model to download
                            def get_model_download_link(pipeline, filename='knn_model.pkl'):
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
        if st.session_state.get('knn_trained_model') is None:
            st.warning("⚠️ You need to train a model first. Go to the 'Train Model' tab to train a KNN model.")
        else:
            # Get model and data from session state
            model = st.session_state.knn_trained_model
            pipeline = st.session_state.knn_pipeline
            X_test = st.session_state.knn_X_test
            y_test = st.session_state.knn_y_test
            feature_names = st.session_state.knn_feature_names
            classes = st.session_state.knn_classes
            metrics = st.session_state.knn_model_metrics
            k_value = metrics["k Value"]
            
            # Create subtabs for different evaluation aspects
            eval_tab1, eval_tab2, eval_tab3 = st.tabs([
                "Model Performance", 
                "Neighbor Analysis", 
                "Error Analysis"
            ])
            
            with eval_tab1:
                st.subheader("Performance Metrics")
                
                # Display metrics in a nice format with descriptions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                    st.caption("Percentage of correct predictions")
                    
                    if len(classes) <= 2:  # Binary classification
                        st.metric("Precision", f"{metrics['Precision']:.4f}")
                        st.caption("Ratio of true positives to all predicted positives")
                    else:  # Multiclass
                        st.metric("Precision (macro)", f"{metrics['Precision (macro)']:.4f}")
                        st.caption("Average precision across all classes")
                
                with col2:
                    if len(classes) <= 2:  # Binary classification
                        st.metric("Recall", f"{metrics['Recall']:.4f}")
                        st.caption("Ratio of true positives to all actual positives")
                        
                        st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                        st.caption("Harmonic mean of precision and recall")
                    else:  # Multiclass
                        st.metric("Recall (macro)", f"{metrics['Recall (macro)']:.4f}")
                        st.caption("Average recall across all classes")
                        
                        st.metric("F1 Score (macro)", f"{metrics['F1 Score (macro)']:.4f}")
                        st.caption("Harmonic mean of macro precision and recall")
                
                with col3:
                    st.metric("k Value", k_value)
                    st.caption("Number of neighbors used for prediction")
                    
                    st.metric("Test Set Size", f"{metrics['Test Set Size']}")
                    st.caption("Number of samples in the test set")
                    
                    st.metric("Training Time", f"{metrics['Training Time']:.2f} sec")
                    st.caption("Time taken to train the model")
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                
                # Get predictions
                y_pred = pipeline.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                
                # Normalize confusion matrix option
                normalize_cm = st.checkbox("Normalize Confusion Matrix", value=False)
                
                if normalize_cm:
                    # Normalize by row (true labels)
                    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    cm_display = cm_norm
                    fmt = '.2f'
                else:
                    cm_display = cm
                    fmt = 'd'
                
                # Create heatmap using plotly for interactivity
                fig = px.imshow(
                    cm_display,
                    labels=dict(x="Predicted Label", y="True Label", color="Count"),
                    x=classes,
                    y=classes,
                    color_continuous_scale="Blues",
                    aspect="equal"
                )
                
                fig.update_layout(
                    title="Confusion Matrix",
                    xaxis_title="Predicted Label",
                    yaxis_title="True Label",
                    width=600,
                    height=600,
                )
                
                # Add text annotations to the heatmap
                for i in range(len(classes)):
                    for j in range(len(classes)):
                        if normalize_cm:
                            text = f"{cm_display[i, j]:.2f}"
                        else:
                            text = f"{cm_display[i, j]}"
                        
                        fig.add_annotation(
                            x=j,
                            y=i,
                            text=text,
                            showarrow=False,
                            font=dict(color="white" if cm_display[i, j] > cm_display.max() / 2 else "black")
                        )
                
                st.plotly_chart(fig)
                
                # Classification Report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                
                # Style the dataframe
                st.dataframe(
                    report_df.style.format({
                        "precision": "{:.3f}",
                        "recall": "{:.3f}",
                        "f1-score": "{:.3f}",
                        "support": "{:.0f}"
                    }),
                    use_container_width=True
                )
            
            with eval_tab2:
                st.subheader("Neighbor Analysis")
                st.info("This tab will provide visualizations and analysis of how neighbors influence predictions.")
            
            with eval_tab3:
                st.subheader("Error Analysis")
                st.info("This tab will provide insights into model errors and potential improvements.")
        
        # Note about model retraining
        st.info("To retrain the model with different parameters, go back to the 'Train Model' tab.")
        
    with tab4:
        st.header("Make Predictions")
        
        # Check if a model has been trained
        if st.session_state.get('knn_trained_model') is None:
            st.warning("⚠️ You need to train a model first. Go to the 'Train Model' tab to train a KNN model.")
        else:
            # Get model and data from session state
            pipeline = st.session_state.knn_pipeline
            feature_names = st.session_state.knn_feature_names
            classes = st.session_state.knn_classes
            
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
                    prediction_probs = pipeline.predict_proba(input_df)[0]
                    
                    # Convert prediction index to class name
                    predicted_class = classes[prediction]
                    
                    # Display prediction
                    st.success(f"### Prediction: {predicted_class}")
                    
                    # Display prediction probabilities
                    st.subheader("Prediction Probabilities")
                    
                    # Create DataFrame for probabilities
                    probs_df = pd.DataFrame({
                        'Class': classes,
                        'Probability': prediction_probs
                    }).sort_values('Probability', ascending=False)
                    
                    # Format as percentages
                    probs_df['Probability'] = probs_df['Probability'] * 100
                    
                    # Display as bar chart
                    fig = px.bar(
                        probs_df,
                        x='Class',
                        y='Probability',
                        title="Prediction Probabilities",
                        labels={'Probability': 'Probability (%)'},
                        color='Probability',
                        color_continuous_scale='Blues'
                    )
                    
                    # Add percentage labels
                    fig.update_traces(
                        texttemplate='%{y:.1f}%',
                        textposition='outside'
                    )
                    
                    # Format y-axis as percentage
                    fig.update_layout(
                        yaxis=dict(
                            ticksuffix='%',
                            range=[0, max(100, probs_df['Probability'].max() * 1.1)]  # Add some padding
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # KNN-specific visualization: nearest neighbors
                    st.subheader("Nearest Neighbors Analysis")
                    
                    try:
                        # Get the KNN model from the pipeline
                        knn_model = st.session_state.knn_trained_model
                        
                        # Preprocess the input data
                        preprocessor = pipeline.named_steps['preprocessor']
                        X_train_processed = preprocessor.transform(st.session_state.knn_X_train)
                        X_input_processed = preprocessor.transform(input_df)
                        
                        # Get nearest neighbors
                        k_value = knn_model.n_neighbors
                        distances, indices = knn_model.kneighbors(X_input_processed)
                        
                        # Get neighbor samples
                        neighbor_indices = indices[0]
                        neighbors_X = st.session_state.knn_X_train.iloc[neighbor_indices]
                        neighbors_y = st.session_state.knn_y_train[neighbor_indices]
                        
                        # Create a dataframe with neighbors info
                        neighbors_df = neighbors_X.copy()
                        neighbors_df['Distance'] = distances[0]
                        neighbors_df['Class'] = [classes[y] for y in neighbors_y]
                        
                        # Sort by distance
                        neighbors_df = neighbors_df.sort_values('Distance')
                        
                        # Show neighbors
                        st.markdown(f"These are the {k_value} nearest neighbors that influenced the prediction:")
                        st.dataframe(neighbors_df, use_container_width=True)
                        
                        # Visualize class distribution in nearest neighbors
                        class_counts = pd.Series([classes[y] for y in neighbors_y]).value_counts()
                        
                        # Create pie chart
                        fig = px.pie(
                            values=class_counts.values,
                            names=class_counts.index,
                            title="Class Distribution in Nearest Neighbors",
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add explanation of how KNN made this prediction
                        most_common_class = class_counts.idxmax()
                        most_common_count = class_counts.max()
                        
                        st.info(f"""
                        **How KNN Made This Prediction**:
                        
                        1. The model found the {k_value} closest neighbors to this data point
                        2. Out of these neighbors, {most_common_count} belong to class "{most_common_class}"
                        3. Therefore, the model predicted the class as "{predicted_class}"
                        
                        In KNN, the prediction is based on a majority vote of the nearest neighbors.
                        """)
                        
                    except Exception as e:
                        st.error(f"Could not analyze nearest neighbors: {str(e)}")
            
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
                    if st.session_state.get('knn_X_test') is not None:
                        X_pred = st.session_state.knn_X_test
                        st.info(f"Using {len(X_pred)} samples from the test set")
                    else:
                        X_pred = df[feature_names]
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
                    X_pred = df[feature_names].sample(n=n_samples, random_state=42)
                    st.info(f"Using {len(X_pred)} random samples from the dataset")
                else:  # Use entire dataset
                    X_pred = df[feature_names]
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
                        predictions = pipeline.predict(X_pred)
                        prediction_probs = pipeline.predict_proba(X_pred)
                        
                        progress_bar.progress(50)
                        status_text.text("Processing results...")
                        
                        # Create results DataFrame
                        results_df = X_pred.copy()
                        results_df['Predicted'] = [classes[p] for p in predictions]
                        
                        # Add probability columns for each class
                        for i, cls in enumerate(classes):
                            results_df[f'Prob_{cls}'] = prediction_probs[:, i]
                        
                        # If actual values are available (from test set)
                        if data_option == "Use test set (if available)" and st.session_state.get('knn_y_test') is not None:
                            y_true = st.session_state.knn_y_test
                            results_df['Actual'] = [classes[y] for y in y_true]
                            results_df['Correct'] = results_df['Predicted'] == results_df['Actual']
                            
                            # Calculate accuracy
                            accuracy = results_df['Correct'].mean() * 100
                            st.metric("Prediction Accuracy", f"{accuracy:.2f}%")
                        
                        progress_bar.progress(100)
                        status_text.text("Predictions complete!")
                        
                        # Display results
                        st.subheader("Prediction Results")
                        st.dataframe(results_df.head(display_limit), use_container_width=True)
                        
                        # Download results
                        st.download_button(
                            "Download All Prediction Results",
                            results_df.to_csv(index=False).encode('utf-8'),
                            "knn_predictions.csv",
                            "text/csv",
                            key='download-batch-predictions'
                        )
                        
                        # Show distribution of predictions
                        st.subheader("Prediction Distribution")
                        
                        # Count predictions by class
                        pred_counts = results_df['Predicted'].value_counts().reset_index()
                        pred_counts.columns = ['Class', 'Count']
                        
                        # Add percentage
                        total_count = pred_counts['Count'].sum()
                        pred_counts['Percentage'] = pred_counts['Count'] / total_count * 100
                        
                        # Create bar chart
                        fig = px.bar(
                            pred_counts,
                            x='Class',
                            y='Count',
                            title="Distribution of Predictions",
                            text='Percentage',
                            color='Count',
                            color_continuous_scale='Blues'
                        )
                        
                        # Add percentage labels
                        fig.update_traces(
                            texttemplate='%{text:.1f}%',
                            textposition='outside'
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
                            
                            # Make predictions button
                            if st.button("Generate Predictions from File"):
                                # Show progress
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                try:
                                    status_text.text("Making predictions...")
                                    
                                    # Make predictions
                                    predictions = pipeline.predict(X_upload)
                                    prediction_probs = pipeline.predict_proba(X_upload)
                                    
                                    progress_bar.progress(50)
                                    status_text.text("Processing results...")
                                    
                                    # Create results DataFrame (include all original columns)
                                    results_df = predict_df.copy()
                                    results_df['Predicted'] = [classes[p] for p in predictions]
                                    
                                    # Add probability columns for each class
                                    for i, cls in enumerate(classes):
                                        results_df[f'Prob_{cls}'] = prediction_probs[:, i]
                                    
                                    progress_bar.progress(100)
                                    status_text.text("Predictions complete!")
                                    
                                    # Display results
                                    st.subheader("Prediction Results")
                                    st.dataframe(results_df.head(display_limit), use_container_width=True)
                                    
                                    # Download results
                                    st.download_button(
                                        "Download All Prediction Results",
                                        results_df.to_csv(index=False).encode('utf-8'),
                                        "knn_file_predictions.csv",
                                        "text/csv",
                                        key='download-file-predictions'
                                    )
                                    
                                    # Show distribution of predictions
                                    st.subheader("Prediction Distribution")
                                    
                                    # Count predictions by class
                                    pred_counts = results_df['Predicted'].value_counts().reset_index()
                                    pred_counts.columns = ['Class', 'Count']
                                    
                                    # Add percentage
                                    total_count = pred_counts['Count'].sum()
                                    pred_counts['Percentage'] = pred_counts['Count'] / total_count * 100
                                    
                                    # Create bar chart
                                    fig = px.bar(
                                        pred_counts,
                                        x='Class',
                                        y='Count',
                                        title="Distribution of Predictions",
                                        text='Percentage',
                                        color='Count',
                                        color_continuous_scale='Blues'
                                    )
                                    
                                    # Add percentage labels
                                    fig.update_traces(
                                        texttemplate='%{text:.1f}%',
                                        textposition='outside'
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
                    
                    No target column is needed since we're making predictions.
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
                   with open('knn_model.pkl', 'rb') as f:
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
                with open('knn_model.pkl', 'rb') as f:
                    model = pickle.load(f)
                
                @app.route('/predict', methods=['POST'])
                def predict():
                    # Get JSON data from request
                    data = request.get_json()
                    
                    # Convert to DataFrame
                    input_df = pd.DataFrame(data, index=[0])
                    
                    # Make prediction
                    prediction = model.predict(input_df)[0]
                    probabilities = model.predict_proba(input_df)[0].tolist()
                    
                    # Return result
                    return jsonify({
                        'prediction': prediction,
                        'probabilities': probabilities
                    })
                
                if __name__ == '__main__':
                    app.run(debug=True)
                ```
                """)

# Footer
create_footer() 
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
        st.info("Train a model first using the 'Train Model' tab.")
    
    with tab4:
        st.header("Make Predictions")
        st.info("Train a model first using the 'Train Model' tab.")
        
else:
    show_file_required_warning()

# Footer
create_footer() 
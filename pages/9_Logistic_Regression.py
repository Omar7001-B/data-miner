import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix, 
    classification_report,
    roc_curve,
    precision_recall_curve
)
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
import pickle
import time
import warnings
import io
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
st.set_page_config(page_title="Logistic Regression", page_icon="🧮", layout="wide")
load_css()

# Main title
st.title("Logistic Regression Classifier 🧮")
st.markdown("""
Train a logistic regression model to classify data. This page helps you set up, train, 
evaluate, and visualize a logistic regression model for your classification problems.
""")

# Check if data is loaded
df = st.session_state.get('df', None)
if df is not None:
    # Display dataset info
    display_dataset_info()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Train Model", "Evaluate Model", "Make Predictions"])
    
    with tab1:
        st.header("Logistic Regression Overview")
        
        st.markdown("""
        ### What is Logistic Regression?
        
        Logistic regression is a statistical model that uses a logistic function to model a binary dependent variable. 
        It's commonly used for classification problems where you want to predict categories like:
        
        - Email spam detection (spam/not spam)
        - Medical diagnosis (disease/no disease)
        - Customer churn prediction (will churn/won't churn)
        - Credit risk assessment (good/bad credit risk)
        
        ### When to Use Logistic Regression
        
        Logistic regression works best when:
        - The relationship between features and outcome is approximately linear
        - You need a model that's easy to interpret
        - You want probability scores, not just classifications
        - The classes in your target variable are separable
        
        ### Advantages of Logistic Regression
        
        - **Interpretability**: Coefficients directly indicate feature importance and direction
        - **Efficiency**: Fast to train, even on large datasets
        - **Probabilistic**: Provides probability scores, not just classifications
        - **Regularization**: Can handle correlated features with L1/L2 regularization
        
        ### Limitations
        
        - Assumes a linear relationship between features and log-odds
        - May underperform with highly non-linear data
        - Can struggle with imbalanced datasets without proper adjustments
        - Not ideal for complex relationships where tree-based models might excel
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
            st.info("No categorical columns detected in this dataset. To use logistic regression, you need a categorical target variable.")
            
    with tab2:
        st.header("Train Logistic Regression Model")
        
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
        if 'lr_classes' not in st.session_state:
            st.session_state.lr_classes = None
        if 'X_train' not in st.session_state:
            st.session_state.X_train = None
        if 'X_test' not in st.session_state:
            st.session_state.X_test = None
        if 'y_train' not in st.session_state:
            st.session_state.y_train = None
        if 'y_test' not in st.session_state:
            st.session_state.y_test = None
        
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
            st.session_state.lr_target = target_col
            
            # Show distribution of target variable
            target_counts = df[target_col].value_counts()
            
            # Create target distribution chart
            fig = px.pie(
                values=target_counts.values,
                names=target_counts.index,
                title=f"Distribution of {target_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Check for class imbalance
            if len(target_counts) >= 2:
                most_common = target_counts.iloc[0]
                least_common = target_counts.iloc[-1]
                imbalance_ratio = most_common / least_common
                
                if imbalance_ratio > 10:
                    st.warning(f"⚠️ Severe class imbalance detected (ratio: {imbalance_ratio:.1f}). This may affect model performance.")
                    st.info("Consider using class weights or data augmentation techniques for better results.")
                elif imbalance_ratio > 3:
                    st.warning(f"⚠️ Moderate class imbalance detected (ratio: {imbalance_ratio:.1f}). Consider using class weights.")
            
            # Step 2: Select features
            st.subheader("Step 2: Select Features")
            
            # Get all columns except target
            potential_features = [col for col in df.columns if col != target_col]
            
            # Allow selecting all or specific features
            select_all = st.checkbox("Select all features", value=True)
            
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
                    # Model configuration options
                    penalty = st.selectbox(
                        "Regularization type:",
                        options=["none", "l2", "l1", "elasticnet"],
                        index=1,
                        help="Type of regularization to prevent overfitting. 'l2' is Ridge, 'l1' is Lasso, 'elasticnet' is a mix of both."
                    )
                    
                    if penalty != "none":
                        C = st.slider(
                            "Inverse of regularization strength (C):",
                            min_value=0.01,
                            max_value=10.0,
                            value=1.0,
                            step=0.01,
                            help="Smaller values specify stronger regularization."
                        )
                    else:
                        C = None
                    
                    # Only show l1_ratio if elasticnet is selected
                    if penalty == "elasticnet":
                        l1_ratio = st.slider(
                            "L1 ratio for elasticnet:",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.5,
                            step=0.1,
                            help="Mix between L1 and L2. 0 = L2, 1 = L1"
                        )
                    else:
                        l1_ratio = None
                    
                    solver = st.selectbox(
                        "Solver algorithm:",
                        options=["lbfgs", "newton-cg", "liblinear", "sag", "saga"],
                        index=0,
                        help="Algorithm to use in the optimization. Some solvers only support certain penalty types."
                    )
                    
                    max_iter = st.slider(
                        "Maximum iterations:",
                        min_value=100,
                        max_value=2000,
                        value=1000,
                        step=100,
                        help="Maximum number of iterations for the solver to converge."
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
                    
                    scale_features = st.checkbox(
                        "Scale numeric features",
                        value=True,
                        help="Standardize numeric features to have mean=0 and variance=1. Recommended for logistic regression."
                    )
                    
                    handle_imbalance = st.checkbox(
                        "Use class weights",
                        value=True,
                        help="Adjust weights inversely proportional to class frequencies to handle imbalanced data."
                    )
                    
                    n_jobs = st.slider(
                        "Number of CPU cores to use:",
                        min_value=-1,
                        max_value=8,
                        value=-1,
                        step=1,
                        help="Number of CPU cores to use. -1 means using all processors."
                    )
                
                # Check solver and penalty compatibility
                if (penalty == "l1" and solver not in ["liblinear", "saga"]) or \
                   (penalty == "elasticnet" and solver != "saga"):
                    st.error(f"The {solver} solver doesn't support {penalty} penalty. Please select a compatible combination.")
                else:
                    # Step 4: Train the model
                    st.subheader("Step 4: Train the Model")
                    
                    train_button = st.button("Train Logistic Regression Model")
                    
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
                                    numeric_transformer = Pipeline(steps=[
                                        ('imputer', SimpleImputer(strategy='median')),
                                        ('scaler', StandardScaler() if scale_features else 'passthrough')
                                    ])
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
                                
                                # Set class_weight parameter
                                if handle_imbalance:
                                    class_weight = 'balanced'
                                else:
                                    class_weight = None
                                
                                # Create model
                                if penalty == 'none':
                                    model = LogisticRegression(
                                        penalty=None,  # None instead of 'none' for scikit-learn
                                        solver=solver,
                                        max_iter=max_iter,
                                        random_state=int(random_state),
                                        class_weight=class_weight,
                                        n_jobs=n_jobs
                                    )
                                elif penalty == 'elasticnet':
                                    model = LogisticRegression(
                                        penalty=penalty,
                                        C=C,
                                        solver=solver,
                                        l1_ratio=l1_ratio,
                                        max_iter=max_iter,
                                        random_state=int(random_state),
                                        class_weight=class_weight,
                                        n_jobs=n_jobs
                                    )
                                else:
                                    model = LogisticRegression(
                                        penalty=penalty,
                                        C=C,
                                        solver=solver,
                                        max_iter=max_iter,
                                        random_state=int(random_state),
                                        class_weight=class_weight,
                                        n_jobs=n_jobs
                                    )
                                
                                # Create full pipeline
                                pipeline = Pipeline(steps=[
                                    ('preprocessor', preprocessor),
                                    ('classifier', model)
                                ])
                                
                                progress_bar.progress(30)
                                status_text.text("Splitting data into train and test sets...")
                                
                                # Split data
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y_encoded, 
                                    test_size=test_size, 
                                    random_state=int(random_state),
                                    stratify=y_encoded  # Ensure balanced classes in train and test sets
                                )
                                
                                progress_bar.progress(40)
                                status_text.text("Training model...")
                                
                                # Train model
                                start_time = time.time()
                                pipeline.fit(X_train, y_train)
                                training_time = time.time() - start_time
                                
                                progress_bar.progress(70)
                                status_text.text("Evaluating model...")
                                
                                # Make predictions
                                y_pred = pipeline.predict(X_test)
                                y_pred_proba = pipeline.predict_proba(X_test)
                                
                                # For multiclass problems, compute class-specific metrics
                                classes = le.classes_
                                is_multiclass = len(classes) > 2
                                
                                # Calculate metrics
                                metrics = {
                                    "Accuracy": accuracy_score(y_test, y_pred),
                                    "Training Time": training_time,
                                    "Test Set Size": len(X_test)
                                }
                                
                                # Binary classification metrics
                                if not is_multiclass:
                                    metrics.update({
                                        "Precision": precision_score(y_test, y_pred, average='binary'),
                                        "Recall": recall_score(y_test, y_pred, average='binary'),
                                        "F1 Score": f1_score(y_test, y_pred, average='binary'),
                                        "ROC AUC": roc_auc_score(y_test, y_pred_proba[:, 1])
                                    })
                                else:
                                    # Multiclass metrics
                                    metrics.update({
                                        "Precision (macro)": precision_score(y_test, y_pred, average='macro'),
                                        "Recall (macro)": recall_score(y_test, y_pred, average='macro'),
                                        "F1 Score (macro)": f1_score(y_test, y_pred, average='macro')
                                    })
                                    
                                    try:
                                        metrics["ROC AUC (macro)"] = roc_auc_score(
                                            y_test, y_pred_proba, 
                                            multi_class='ovr', 
                                            average='macro'
                                        )
                                    except:
                                        metrics["ROC AUC (macro)"] = "N/A"
                                
                                progress_bar.progress(90)
                                
                                # Save to session state
                                st.session_state.lr_trained_model = model
                                st.session_state.lr_model_metrics = metrics
                                st.session_state.lr_pipeline = pipeline
                                st.session_state.lr_feature_names = selected_features
                                st.session_state.lr_classes = classes
                                st.session_state.X_train = X_train
                                st.session_state.X_test = X_test
                                st.session_state.y_train = y_train
                                st.session_state.y_test = y_test
                                
                                progress_bar.progress(100)
                                status_text.text("Training complete!")
                                
                                # Display results
                                st.success("✅ Model training complete!")
                                
                                # Display metrics in a nice format
                                st.subheader("Model Performance Metrics")
                                
                                # Format metrics for display
                                metric_cols = st.columns(4)
                                
                                with metric_cols[0]:
                                    st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                                
                                with metric_cols[1]:
                                    if not is_multiclass:
                                        st.metric("Precision", f"{metrics['Precision']:.4f}")
                                    else:
                                        st.metric("Precision (macro)", f"{metrics['Precision (macro)']:.4f}")
                                
                                with metric_cols[2]:
                                    if not is_multiclass:
                                        st.metric("Recall", f"{metrics['Recall']:.4f}")
                                    else:
                                        st.metric("Recall (macro)", f"{metrics['Recall (macro)']:.4f}")
                                
                                with metric_cols[3]:
                                    if not is_multiclass:
                                        st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                                    else:
                                        st.metric("F1 Score (macro)", f"{metrics['F1 Score (macro)']:.4f}")
                                
                                # Display ROC AUC for binary case
                                if not is_multiclass:
                                    st.metric("ROC AUC", f"{metrics['ROC AUC']:.4f}")
                                elif "ROC AUC (macro)" in metrics and metrics["ROC AUC (macro)"] != "N/A":
                                    st.metric("ROC AUC (macro)", f"{metrics['ROC AUC (macro)']:.4f}")
                                
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
                                def get_model_download_link(pipeline, filename='logistic_regression_model.pkl'):
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
        
        # Check if model is trained
        if st.session_state.lr_trained_model is None:
            st.warning("Please train a model first in the 'Train Model' tab.")
        else:
            # Get model and results from session state
            model = st.session_state.lr_trained_model
            pipeline = st.session_state.lr_pipeline
            metrics = st.session_state.lr_model_metrics
            classes = st.session_state.lr_classes
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            feature_names = st.session_state.lr_feature_names
            
            # Create tabs for different evaluations
            eval_tab1, eval_tab2, eval_tab3 = st.tabs(["Model Performance", "Feature Importance", "Decision Boundary"])
            
            with eval_tab1:
                st.subheader("Model Performance Metrics")
                
                # Fix: Check if classes is None before using len(classes)
                if classes is None:
                    is_multiclass = False
                    st.warning("Class labels are missing. Some metrics and plots may not display correctly. Please retrain the model if this persists.")
                    classes = ["Class 0", "Class 1"]  # Fallback for binary
                else:
                    is_multiclass = len(classes) > 2
                
                # Display metrics in a nice format
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                
                with metric_cols[1]:
                    if not is_multiclass:
                        st.metric("Precision", f"{metrics['Precision']:.4f}")
                    else:
                        st.metric("Precision (macro)", f"{metrics['Precision (macro)']:.4f}")
                
                with metric_cols[2]:
                    if not is_multiclass:
                        st.metric("Recall", f"{metrics['Recall']:.4f}")
                    else:
                        st.metric("Recall (macro)", f"{metrics['Recall (macro)']:.4f}")
                
                with metric_cols[3]:
                    if not is_multiclass:
                        st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                    else:
                        st.metric("F1 Score (macro)", f"{metrics['F1 Score (macro)']:.4f}")
                
                # Show training details
                st.markdown(f"**Training Time:** {metrics['Training Time']:.2f} seconds")
                st.markdown(f"**Test Set Size:** {metrics['Test Set Size']} samples")
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                
                y_pred = pipeline.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                
                # Normalize confusion matrix option
                normalize_cm = st.checkbox("Normalize Confusion Matrix", value=False)
                
                if normalize_cm:
                    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(
                        cm_normalized, 
                        annot=True, 
                        fmt='.2f', 
                        cmap='Blues',
                        xticklabels=classes,
                        yticklabels=classes
                    )
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    plt.title('Normalized Confusion Matrix')
                else:
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
                
                # Binary classification: ROC curve
                if not is_multiclass:
                    st.subheader("ROC Curve")
                    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic (ROC) Curve')
                    plt.legend(loc="lower right")
                    st.pyplot(fig)
                    
                    # Precision-Recall curve
                    st.subheader("Precision-Recall Curve")
                    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    plt.plot(recall, precision, label=f'Precision-Recall curve')
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.title('Precision-Recall Curve')
                    plt.ylim([0.0, 1.05])
                    plt.xlim([0.0, 1.0])
                    plt.legend(loc="lower left")
                    st.pyplot(fig)
                else:
                    # For multiclass, show ROC curves for each class (one-vs-rest)
                    st.subheader("ROC Curves (One-vs-Rest)")
                    
                    y_pred_proba = pipeline.predict_proba(X_test)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    for i, class_name in enumerate(classes):
                        # Binarize the target for the current class (one-vs-rest)
                        y_test_binary = (y_test == i).astype(int)
                        
                        # Get ROC curve
                        try:
                            fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba[:, i])
                            roc_auc = roc_auc_score(y_test_binary, y_pred_proba[:, i])
                            
                            plt.plot(fpr, tpr, lw=2,
                                     label=f'Class {class_name} (area = {roc_auc:.3f})')
                        except Exception as e:
                            st.warning(f"Could not compute ROC curve for class {class_name}: {str(e)}")
                    
                    plt.plot([0, 1], [0, 1], 'k--', lw=2)
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curves (One-vs-Rest)')
                    plt.legend(loc="lower right")
                    
                    st.pyplot(fig)
                
                # Classification report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
            
            with eval_tab2:
                st.subheader("Feature Importance")
                
                # Get the trained logistic regression model from the pipeline
                try:
                    # Get feature names after preprocessing
                    preprocessor = pipeline.named_steps['preprocessor']
                    
                    # Try to get feature names from OneHotEncoder
                    try:
                        if hasattr(preprocessor, 'get_feature_names_out'):
                            feature_names_after_preprocessing = preprocessor.get_feature_names_out()
                        else:
                            feature_names_after_preprocessing = None
                            st.warning("Could not determine feature names after preprocessing. Using default names.")
                    except Exception as e:
                        feature_names_after_preprocessing = None
                        st.warning(f"Error getting feature names: {str(e)}")
                    
                    # Get coefficients
                    model = pipeline.named_steps['classifier']
                    coefficients = model.coef_
                    
                    # For binary classification, coefficients are in a single row
                    if not is_multiclass:
                        if feature_names_after_preprocessing is not None:
                            features_df = pd.DataFrame({
                                'Feature': feature_names_after_preprocessing,
                                'Coefficient': coefficients[0]
                            })
                        else:
                            # Use generic feature names if actual names can't be determined
                            features_df = pd.DataFrame({
                                'Feature': [f"Feature {i}" for i in range(len(coefficients[0]))],
                                'Coefficient': coefficients[0]
                            })
                        
                        # Sort by absolute value of coefficient
                        features_df['Abs_Coefficient'] = np.abs(features_df['Coefficient'])
                        features_df = features_df.sort_values('Abs_Coefficient', ascending=False)
                        
                        # Bar chart of coefficients
                        st.markdown("The coefficients in logistic regression indicate the change in log odds for a one-unit "
                                   "change in the predictor. Larger absolute values indicate stronger influence.")
                        
                        # Show table of coefficients
                        st.dataframe(features_df[['Feature', 'Coefficient']].head(20))
                        
                        # Create horizontal bar chart of top features
                        top_n = st.slider("Number of top features to display:", 5, 20, 10)
                        top_features = features_df.head(top_n)
                        
                        fig = px.bar(
                            top_features,
                            x='Coefficient',
                            y='Feature',
                            orientation='h',
                            title=f'Top {top_n} Feature Coefficients',
                            color='Coefficient',
                            color_continuous_scale='RdBu_r'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # For multiclass, we have one row of coefficients per class
                        st.markdown("For multiclass logistic regression, we have one set of coefficients per class. "
                                   "Each coefficient represents the importance of a feature for predicting that specific class.")
                        
                        # Select class to show coefficients for
                        selected_class = st.selectbox(
                            "Select class to show coefficients for:",
                            options=list(range(len(classes))),
                            format_func=lambda i: classes[i]
                        )
                        
                        if feature_names_after_preprocessing is not None:
                            features_df = pd.DataFrame({
                                'Feature': feature_names_after_preprocessing,
                                'Coefficient': coefficients[selected_class]
                            })
                        else:
                            # Use generic feature names
                            features_df = pd.DataFrame({
                                'Feature': [f"Feature {i}" for i in range(len(coefficients[selected_class]))],
                                'Coefficient': coefficients[selected_class]
                            })
                        
                        # Sort by absolute value
                        features_df['Abs_Coefficient'] = np.abs(features_df['Coefficient'])
                        features_df = features_df.sort_values('Abs_Coefficient', ascending=False)
                        
                        # Show table
                        st.dataframe(features_df[['Feature', 'Coefficient']].head(20))
                        
                        # Bar chart
                        top_n = st.slider("Number of top features to display:", 5, 20, 10)
                        top_features = features_df.head(top_n)
                        
                        fig = px.bar(
                            top_features,
                            x='Coefficient',
                            y='Feature',
                            orientation='h',
                            title=f'Top {top_n} Feature Coefficients for Class: {classes[selected_class]}',
                            color='Coefficient',
                            color_continuous_scale='RdBu_r'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"An error occurred while analyzing feature importance: {str(e)}")
                    st.info("This may happen with complex preprocessing pipelines or if the model structure doesn't support coefficient extraction.")
            
            with eval_tab3:
                st.subheader("Decision Boundary Visualization")
                
                # For decision boundary, we need to reduce to 2D if we have more than 2 features
                st.markdown("""
                To visualize the decision boundary, we need to reduce the feature space to 2 dimensions.
                This is a simplified view but helps to understand how the model separates classes.
                """)
                
                # Check if we have at least 2 numeric features
                numeric_features = get_numeric_columns(pd.DataFrame(X_test, columns=feature_names))
                
                if len(numeric_features) < 2:
                    st.warning("Need at least 2 numeric features to visualize decision boundary. Your dataset doesn't have enough numeric features.")
                else:
                    # Allow user to select two features to visualize
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        feature1 = st.selectbox(
                            "Select first feature for visualization:",
                            options=numeric_features,
                            index=0
                        )
                    
                    with col2:
                        # Default to second feature, but avoid duplicate selection
                        default_idx = 1 if len(numeric_features) > 1 else 0
                        default_idx = min(default_idx, len(numeric_features) - 1)
                        
                        feature2 = st.selectbox(
                            "Select second feature for visualization:",
                            options=numeric_features,
                            index=default_idx
                        )
                    
                    if feature1 == feature2:
                        st.warning("Please select two different features for meaningful visualization.")
                    else:
                        try:
                            # Get indices of the two selected features
                            idx1 = feature_names.index(feature1)
                            idx2 = feature_names.index(feature2)
                            
                            # Extract just the two features for visualization
                            X_test_2d = X_test.copy()
                            X_test_2d = pd.DataFrame(X_test_2d, columns=feature_names)
                            X_test_2d = X_test_2d[[feature1, feature2]]
                            
                            # Create a meshgrid to visualize decision boundary
                            x_min, x_max = X_test_2d[feature1].min() - 0.5, X_test_2d[feature1].max() + 0.5
                            y_min, y_max = X_test_2d[feature2].min() - 0.5, X_test_2d[feature2].max() + 0.5
                            
                            # Reduce meshgrid density for large ranges to improve performance
                            x_range = x_max - x_min
                            y_range = y_max - y_min
                            
                            # Adjust mesh density based on data range
                            if x_range * y_range > 1000:
                                mesh_density = 50  # Lower density for large ranges
                            else:
                                mesh_density = 100
                            
                            xx, yy = np.meshgrid(
                                np.linspace(x_min, x_max, mesh_density),
                                np.linspace(y_min, y_max, mesh_density)
                            )
                            
                            # Create a simplified pipeline with just the two features
                            # Since we can't directly use our pipeline (it expects all original features),
                            # we'll create a new, simpler model fit on just these two features
                            
                            # Train a simpler model on just the two features
                            simple_model = LogisticRegression(
                                max_iter=1000,
                                random_state=42
                            )
                            
                            # Keep the original preprocessing for just these two columns
                            X_train_2d = pd.DataFrame(st.session_state.X_train, columns=feature_names)
                            X_train_2d = X_train_2d[[feature1, feature2]]
                            
                            # Fit the simple model
                            simple_model.fit(X_train_2d, st.session_state.y_train)
                            
                            # Use the model to predict on the meshgrid
                            Z = simple_model.predict(np.c_[xx.ravel(), yy.ravel()])
                            Z = Z.reshape(xx.shape)
                            
                            # Create plot
                            fig, ax = plt.subplots(figsize=(10, 8))
                            
                            # Plot decision boundary
                            ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
                            
                            # Plot test points
                            scatter = ax.scatter(
                                X_test_2d[feature1], 
                                X_test_2d[feature2], 
                                c=y_test, 
                                edgecolors='k',
                                cmap='coolwarm'
                            )
                            
                            # Add legend
                            legend1 = ax.legend(
                                scatter.legend_elements()[0], 
                                classes,
                                title="Classes"
                            )
                            ax.add_artist(legend1)
                            
                            ax.set_xlim(xx.min(), xx.max())
                            ax.set_ylim(yy.min(), yy.max())
                            ax.set_xlabel(feature1)
                            ax.set_ylabel(feature2)
                            ax.set_title('Decision Boundary')
                            
                            st.pyplot(fig)
                            
                            st.caption("""
                            Note: This is a simplified visualization based on just two features. 
                            The actual model uses all selected features and may have different decision boundaries.
                            """)
                            
                        except Exception as e:
                            st.error(f"Error creating decision boundary visualization: {str(e)}")
                            st.info("This may happen if the selected features can't be easily visualized or if there are issues with the data.")
    with tab4:
        st.header("Make Predictions")
        
        # Check if model is trained
        if st.session_state.lr_trained_model is None:
            st.warning("Please train a model first in the 'Train Model' tab.")
        else:
            # Get model and related info from session state
            pipeline = st.session_state.lr_pipeline
            feature_names = st.session_state.lr_feature_names
            classes = st.session_state.lr_classes
            
            st.markdown("""
            Use the trained logistic regression model to make predictions on new data. You can:
            1. Make predictions on individual samples by entering values manually
            2. Make predictions on a batch of samples from the test set
            3. Upload new data for prediction
            """)
            
            # Create tabs for different prediction methods
            pred_tab1, pred_tab2, pred_tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Upload & Predict"])
            
            with pred_tab1:
                st.subheader("Make Single Prediction")
                st.markdown("Enter values for each feature to get a prediction.")
                
                # Create a form for input
                with st.form(key="prediction_form"):
                    # Create input fields for each feature
                    feature_values = {}
                    
                    # Get numeric and categorical columns
                    numeric_features = get_numeric_columns(df[feature_names])
                    categorical_features = [col for col in feature_names if col not in numeric_features]
                    
                    # Display inputs by type
                    if numeric_features:
                        st.subheader("Numeric Features")
                        
                        # Calculate min, max, mean for each numeric feature for better defaults
                        feature_stats = {}
                        for feat in numeric_features:
                            feature_stats[feat] = {
                                'min': float(df[feat].min()),
                                'max': float(df[feat].max()),
                                'mean': float(df[feat].mean())
                            }
                        
                        # Create columns for better layout with many features
                        num_cols = 2
                        numeric_rows = [numeric_features[i:i+num_cols] for i in range(0, len(numeric_features), num_cols)]
                        
                        for row in numeric_rows:
                            cols = st.columns(num_cols)
                            for i, feat in enumerate(row):
                                with cols[i]:
                                    min_val = feature_stats[feat]['min']
                                    max_val = feature_stats[feat]['max']
                                    mean_val = feature_stats[feat]['mean']
                                    
                                    # Use number input for numeric features
                                    feature_values[feat] = st.number_input(
                                        f"{feat}:",
                                        min_value=float(min_val - abs(min_val * 0.5)),
                                        max_value=float(max_val + abs(max_val * 0.5)),
                                        value=float(mean_val),
                                        format="%.5f"
                                    )
                    
                    if categorical_features:
                        st.subheader("Categorical Features")
                        
                        # Create columns for better layout with many features
                        cat_cols = 2
                        categorical_rows = [categorical_features[i:i+cat_cols] for i in range(0, len(categorical_features), cat_cols)]
                        
                        for row in categorical_rows:
                            cols = st.columns(cat_cols)
                            for i, feat in enumerate(row):
                                with cols[i]:
                                    # Get unique values for the categorical feature
                                    unique_values = df[feat].dropna().unique().tolist()
                                    default_value = unique_values[0] if unique_values else ""
                                    
                                    # Use selectbox for categorical features
                                    feature_values[feat] = st.selectbox(
                                        f"{feat}:",
                                        options=unique_values,
                                        index=0
                                    )
                    
                    # Submit button
                    submit_button = st.form_submit_button(label="Make Prediction")
                
                # Make prediction when form is submitted
                if submit_button:
                    try:
                        # Create a single sample dataframe from the input values
                        sample = pd.DataFrame([feature_values])
                        
                        # Make prediction
                        prediction = pipeline.predict(sample)[0]
                        prediction_proba = pipeline.predict_proba(sample)[0]
                        
                        # Show results in a nice format
                        st.success(f"### Prediction: {classes[prediction]}")
                        
                        # Show prediction probabilities
                        st.subheader("Prediction Probabilities")
                        
                        # Create a dataframe for probabilities
                        proba_df = pd.DataFrame({
                            'Class': classes,
                            'Probability': prediction_proba
                        })
                        proba_df = proba_df.sort_values('Probability', ascending=False)
                        
                        # Display as a bar chart
                        fig = px.bar(
                            proba_df,
                            x='Probability',
                            y='Class',
                            orientation='h',
                            title='Prediction Probabilities',
                            color='Probability',
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display as table
                        st.dataframe(
                            proba_df.style.background_gradient(cmap='Greens', subset=['Probability']),
                            hide_index=True,
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                        st.info("This might happen if the input data format doesn't match what the model expects.")
            
            with pred_tab2:
                st.subheader("Batch Prediction on Test Data")
                
                # Get test set from session state
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                
                st.markdown(f"Make predictions on samples from the test set (total: {len(X_test)} samples).")
                
                # Allow user to select number of samples
                num_samples = st.slider(
                    "Number of random samples to predict:",
                    min_value=1,
                    max_value=min(100, len(X_test)),
                    value=10
                )
                
                if st.button("Run Batch Prediction"):
                    # Randomly select samples
                    indices = np.random.choice(len(X_test), size=num_samples, replace=False)
                    X_sample = pd.DataFrame(X_test[indices], columns=feature_names)
                    y_sample = y_test[indices]
                    
                    # Make predictions
                    try:
                        y_pred = pipeline.predict(X_sample)
                        y_pred_proba = pipeline.predict_proba(X_sample)
                        
                        # Create results dataframe
                        results = pd.DataFrame({
                            'Sample': range(1, num_samples + 1),
                            'True Class': [classes[y] for y in y_sample],
                            'Predicted Class': [classes[y] for y in y_pred],
                            'Correct': y_sample == y_pred
                        })
                        
                        # Add prediction probabilities
                        for i, cls in enumerate(classes):
                            results[f'Prob_{cls}'] = y_pred_proba[:, i]
                        
                        # Display results
                        st.dataframe(
                            results.style.apply(
                                lambda x: ['background-color: #c6ebc9' if v else 'background-color: #ffc1c1' for v in x.Correct], 
                                axis=1, subset=['Correct']
                            ),
                            use_container_width=True
                        )
                        
                        # Show overall accuracy
                        accuracy = (results['Correct'].sum() / num_samples) * 100
                        st.metric("Batch Accuracy", f"{accuracy:.2f}%")
                        
                        # Option to show the features for these samples
                        if st.checkbox("Show feature values for these samples"):
                            st.dataframe(X_sample, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error making batch predictions: {str(e)}")
            
            with pred_tab3:
                st.subheader("Upload and Predict")
                st.markdown("Upload a CSV or Excel file with new data for prediction. The file should contain columns matching the feature names used for training.")
                
                # File uploader
                uploaded_file = st.file_uploader(
                    "Upload data file (CSV or Excel)",
                    type=["csv", "xlsx"]
                )
                
                if uploaded_file is not None:
                    try:
                        # Load the data
                        if uploaded_file.name.endswith('.csv'):
                            new_data = pd.read_csv(uploaded_file)
                        else:
                            new_data = pd.read_excel(uploaded_file)
                        
                        # Display the uploaded data
                        st.subheader("Uploaded Data")
                        st.dataframe(new_data.head(), use_container_width=True)
                        
                        # Check if the required features are present
                        missing_features = [f for f in feature_names if f not in new_data.columns]
                        
                        if missing_features:
                            st.error(f"Missing required features: {', '.join(missing_features)}")
                            st.info(f"The model requires these features: {', '.join(feature_names)}")
                        else:
                            # Extract only the required features
                            X_new = new_data[feature_names].copy()
                            
                            # Check for missing values
                            if X_new.isnull().any().any():
                                st.warning("The uploaded data contains missing values. The model will try to handle them, but results may be affected.")
                            
                            # Make predictions
                            if st.button("Make Predictions on Uploaded Data"):
                                try:
                                    with st.spinner("Making predictions..."):
                                        # Make predictions
                                        predictions = pipeline.predict(X_new)
                                        probabilities = pipeline.predict_proba(X_new)
                                        
                                        # Create results dataframe
                                        results = pd.DataFrame({
                                            'Predicted Class': [classes[p] for p in predictions]
                                        })
                                        
                                        # Add probabilities
                                        for i, cls in enumerate(classes):
                                            results[f'Probability_{cls}'] = probabilities[:, i]
                                        
                                        # Combine with original data
                                        final_results = pd.concat([new_data, results], axis=1)
                                        
                                        # Display results
                                        st.subheader("Prediction Results")
                                        st.dataframe(final_results, use_container_width=True)
                                        
                                        # Option to download results
                                        csv = final_results.to_csv(index=False)
                                        st.download_button(
                                            "Download Results as CSV",
                                            csv,
                                            file_name="prediction_results.csv",
                                            mime="text/csv"
                                        )
                                except Exception as e:
                                    st.error(f"Error making predictions: {str(e)}")
                                    st.info("This might happen if the data format doesn't match what the model expects.")
                    except Exception as e:
                        st.error(f"Error loading the file: {str(e)}")
                        st.info("Please make sure the file is a valid CSV or Excel file.")
else:
    show_file_required_warning()

# Footer
create_footer() 
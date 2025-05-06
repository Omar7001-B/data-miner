import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
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
st.set_page_config(page_title="Decision Tree", page_icon="🌳", layout="wide")
load_css()

# Main title
st.title("Decision Tree Classifier 🌳")
st.markdown("""
Train a decision tree model to classify data. This page helps you set up, train, 
evaluate, and visualize a decision tree model for your classification problems.
""")

# Check if data is loaded
df = st.session_state.get('df', None)
if df is not None:
    # Display dataset info
    display_dataset_info()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Train Model", "Evaluate Model", "Make Predictions"])
    
    with tab1:
        st.header("Decision Tree Overview")
        
        st.markdown("""
        ### What is a Decision Tree?
        
        A Decision Tree is a flowchart-like structure where each internal node represents a feature, 
        each branch represents a decision rule, and each leaf node represents an outcome. It's one 
        of the most interpretable machine learning models.
        
        ### When to Use Decision Trees
        
        Decision trees work well when:
        - You need a model that's easy to interpret and explain
        - The relationship between features and target is potentially non-linear
        - You're dealing with categorical or numerical features
        - You want to visualize decision-making logic
        
        ### Advantages of Decision Trees
        
        - **Interpretability**: Can be visualized and easily explained
        - **Handles mixed data**: Works with both categorical and numerical features
        - **Non-parametric**: Makes no assumptions about data distribution
        - **Automatic feature selection**: Focuses on important features
        
        ### Limitations
        
        - Tendency to overfit without proper pruning
        - Can be unstable (small changes in data can lead to large changes in tree structure)
        - May create biased trees if classes are imbalanced
        - Not ideal for capturing complex relationships that require many features
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
            st.info("No categorical columns detected in this dataset. To use a decision tree classifier, you need a categorical target variable.")
            
    with tab2:
        st.header("Train Decision Tree Model")
        
        # Initialize session state variables for model training
        if 'dt_target' not in st.session_state:
            st.session_state.dt_target = None
        if 'dt_features' not in st.session_state:
            st.session_state.dt_features = []
        if 'dt_trained_model' not in st.session_state:
            st.session_state.dt_trained_model = None
        if 'dt_model_metrics' not in st.session_state:
            st.session_state.dt_model_metrics = None
        if 'dt_pipeline' not in st.session_state:
            st.session_state.dt_pipeline = None
        if 'dt_feature_names' not in st.session_state:
            st.session_state.dt_feature_names = None
        if 'dt_classes' not in st.session_state:
            st.session_state.dt_classes = None
        if 'dt_X_train' not in st.session_state:
            st.session_state.dt_X_train = None
        if 'dt_X_test' not in st.session_state:
            st.session_state.dt_X_test = None
        if 'dt_y_train' not in st.session_state:
            st.session_state.dt_y_train = None
        if 'dt_y_test' not in st.session_state:
            st.session_state.dt_y_test = None
        if 'dt_tree_visualization' not in st.session_state:
            st.session_state.dt_tree_visualization = None
        
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
            st.session_state.dt_target = target_col
            
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
                    default=st.session_state.dt_features if st.session_state.dt_features else []
                )
            
            # Update session state
            st.session_state.dt_features = selected_features
            
            if not selected_features:
                st.warning("Please select at least one feature for training.")
            
            # Step 3: Configure model parameters
            if selected_features:
                st.subheader("Step 3: Configure Model Parameters")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Tree-specific parameters
                    criterion = st.selectbox(
                        "Split criterion:",
                        options=["gini", "entropy", "log_loss"],
                        index=0,
                        help="Function to measure the quality of a split. 'gini' is for Gini impurity, 'entropy' for information gain."
                    )
                    
                    splitter = st.selectbox(
                        "Splitting strategy:",
                        options=["best", "random"],
                        index=0,
                        help="'best' chooses the best split, 'random' chooses the best random split."
                    )
                    
                    max_depth = st.slider(
                        "Maximum tree depth:",
                        min_value=1,
                        max_value=30,
                        value=5,
                        help="Maximum depth of the tree. Deeper trees can model more complex patterns but may overfit."
                    )
                    
                    min_samples_split = st.slider(
                        "Minimum samples required to split:",
                        min_value=2,
                        max_value=20,
                        value=2,
                        help="Minimum number of samples required to split an internal node."
                    )
                    
                    min_samples_leaf = st.slider(
                        "Minimum samples in leaf node:",
                        min_value=1,
                        max_value=20,
                        value=1,
                        help="Minimum number of samples required to be at a leaf node."
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
                    
                    handle_imbalance = st.checkbox(
                        "Use class weights",
                        value=True,
                        help="Adjust weights inversely proportional to class frequencies to handle imbalanced data."
                    )
                    
                    min_impurity_decrease = st.slider(
                        "Minimum impurity decrease:",
                        min_value=0.0,
                        max_value=0.1,
                        value=0.0,
                        step=0.001,
                        format="%.3f",
                        help="A node will be split if this split induces a decrease of the impurity greater than or equal to this value."
                    )
                    
                    max_leaf_nodes = st.number_input(
                        "Maximum leaf nodes:",
                        min_value=0,
                        max_value=100,
                        value=0,
                        help="Maximum number of leaf nodes. 0 means unlimited."
                    )
                
                # Step 4: Train the model
                st.subheader("Step 4: Train the Model")
                
                train_button = st.button("Train Decision Tree Model")
                
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
                                    ('imputer', SimpleImputer(strategy='median'))
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
                            model = DecisionTreeClassifier(
                                criterion=criterion,
                                splitter=splitter,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                min_impurity_decrease=min_impurity_decrease,
                                max_leaf_nodes=max_leaf_nodes if max_leaf_nodes != 0 else None,
                                class_weight=class_weight,
                                random_state=int(random_state)
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
                            
                            # For multiclass problems, compute class-specific metrics
                            classes = le.classes_
                            is_multiclass = len(classes) > 2
                            
                            # Calculate metrics
                            metrics = {
                                "Accuracy": accuracy_score(y_test, y_pred),
                                "Training Time": training_time,
                                "Test Set Size": len(X_test),
                                "Tree Depth": model.get_depth(),
                                "Number of Leaves": model.get_n_leaves()
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
                            
                            progress_bar.progress(80)
                            status_text.text("Generating tree visualization...")
                            
                            # Create tree visualization
                            try:
                                feature_names_after_preprocessing = None
                                if hasattr(preprocessor, 'get_feature_names_out'):
                                    feature_names_after_preprocessing = preprocessor.get_feature_names_out()
                                
                                # Create a simplified tree visualization
                                plt.figure(figsize=(15, 10))
                                _ = plot_tree(
                                    model,
                                    feature_names=feature_names_after_preprocessing,
                                    class_names=[str(c) for c in classes],
                                    filled=True,
                                    rounded=True,
                                    fontsize=10,
                                    max_depth=3,  # Limit depth for visualization
                                    proportion=True
                                )
                                plt.tight_layout()
                                tree_viz_fig = plt.gcf()
                                
                                # Also create a textual representation of the tree
                                tree_text = export_text(
                                    model,
                                    feature_names=feature_names_after_preprocessing.tolist() if feature_names_after_preprocessing is not None else None,
                                    show_weights=True
                                )
                                
                                # Store visualization
                                st.session_state.dt_tree_visualization = {
                                    'figure': tree_viz_fig,
                                    'text': tree_text
                                }
                            except Exception as e:
                                st.warning(f"Could not generate tree visualization: {str(e)}")
                                st.session_state.dt_tree_visualization = None
                            
                            progress_bar.progress(90)
                            
                            # Save to session state
                            st.session_state.dt_trained_model = model
                            st.session_state.dt_model_metrics = metrics
                            st.session_state.dt_pipeline = pipeline
                            st.session_state.dt_feature_names = selected_features
                            st.session_state.dt_classes = classes
                            st.session_state.dt_X_train = X_train
                            st.session_state.dt_X_test = X_test
                            st.session_state.dt_y_train = y_train
                            st.session_state.dt_y_test = y_test
                            
                            progress_bar.progress(100)
                            status_text.text("Training complete!")
                            
                            # Display results
                            st.success("✅ Model training complete!")
                            
                            # Display metrics in a nice format
                            st.subheader("Model Performance Metrics")
                            
                            # Create two rows of metrics
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
                            
                            # Tree-specific metrics
                            metric_cols2 = st.columns(3)
                            
                            with metric_cols2[0]:
                                st.metric("Tree Depth", metrics["Tree Depth"])
                            
                            with metric_cols2[1]:
                                st.metric("Number of Leaves", metrics["Number of Leaves"])
                            
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
                            
                            # Tree visualization preview
                            if st.session_state.dt_tree_visualization is not None:
                                st.subheader("Decision Tree Preview (Simplified)")
                                st.pyplot(st.session_state.dt_tree_visualization['figure'])
                                
                                with st.expander("Tree Structure as Text"):
                                    st.text(st.session_state.dt_tree_visualization['text'])
                                
                                st.info("⚠️ This is a simplified visualization. Go to the 'Evaluate Model' tab for more detailed tree analysis.")
                            
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
                            def get_model_download_link(pipeline, filename='decision_tree_model.pkl'):
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
        if st.session_state.get('dt_trained_model') is None:
            st.warning("⚠️ You need to train a model first. Go to the 'Train Model' tab to train a Decision Tree model.")
        else:
            # Get model and data from session state
            model = st.session_state.dt_trained_model
            pipeline = st.session_state.dt_pipeline
            X_test = st.session_state.dt_X_test
            y_test = st.session_state.dt_y_test
            feature_names = st.session_state.dt_feature_names
            classes = st.session_state.dt_classes
            metrics = st.session_state.dt_model_metrics
            
            # Create subtabs for different evaluation aspects
            eval_tab1, eval_tab2, eval_tab3, eval_tab4 = st.tabs([
                "Model Performance", 
                "Tree Visualization", 
                "Feature Importance",
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
                    st.metric("Tree Depth", metrics["Tree Depth"])
                    st.caption("Maximum depth of the decision tree")
                    
                    st.metric("Number of Leaves", metrics["Number of Leaves"])
                    st.caption("Total number of leaf nodes in the tree")
                    
                    st.metric("Training Time", f"{metrics['Training Time']:.2f} sec")
                    st.caption("Time taken to train the model")
                
                # ROC Curve for binary classification
                if len(classes) == 2:
                    st.subheader("ROC Curve")
                    
                    from sklearn.metrics import roc_curve, auc, roc_auc_score
                    
                    # Get predictions
                    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                    
                    # Calculate ROC curve and AUC
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    # Plot ROC curve
                    fig = plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic (ROC) Curve')
                    plt.legend(loc="lower right")
                    
                    st.pyplot(fig)
                    
                    st.info("""
                    **Understanding the ROC Curve**:
                    - A ROC curve plots True Positive Rate against False Positive Rate at various threshold settings
                    - The area under the ROC curve (AUC) is a measure of classifier performance
                    - An AUC of 1.0 represents a perfect classifier, while 0.5 represents a classifier no better than random guessing
                    """)
                
                # Confusion Matrix (interactive)
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
                
                st.info("""
                **Understanding the Confusion Matrix**:
                - Each row represents the actual class, while each column represents the predicted class
                - The diagonal elements represent correct predictions
                - Off-diagonal elements represent misclassifications
                - A good model will have high values along the diagonal and low values elsewhere
                """)
                
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
                
                st.info("""
                **Classification Report Explained**:
                - **Precision**: Ratio of true positives to all predicted positives
                - **Recall**: Ratio of true positives to all actual positives
                - **F1-score**: Harmonic mean of precision and recall
                - **Support**: Number of occurrences of each class in the test set
                """)
            
            with eval_tab2:
                st.subheader("Decision Tree Visualization")
                
                # Get tree visualization from session state
                tree_viz = st.session_state.get('dt_tree_visualization')
                
                if tree_viz is None:
                    st.warning("Tree visualization could not be generated.")
                else:
                    # Visualization controls
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        max_depth_viz = st.slider(
                            "Max depth to display:",
                            min_value=1,
                            max_value=model.get_depth(),
                            value=min(3, model.get_depth()),
                            help="Control how many levels of the tree to display"
                        )
                    
                    with col2:
                        filled = st.checkbox("Color-filled nodes", value=True, help="Color nodes by majority class")
                        rounded = st.checkbox("Rounded corners", value=True, help="Use rounded rectangles for nodes")
                    
                    # Generate tree visualization with user settings
                    plt.figure(figsize=(15, 10))
                    
                    # Extract feature names from pipeline for proper labeling
                    feature_names_after_preprocessing = None
                    preprocessor = pipeline.named_steps.get('preprocessor')
                    if preprocessor and hasattr(preprocessor, 'get_feature_names_out'):
                        try:
                            feature_names_after_preprocessing = preprocessor.get_feature_names_out()
                        except:
                            pass
                    
                    _ = plot_tree(
                        model,
                        feature_names=feature_names_after_preprocessing,
                        class_names=[str(c) for c in classes],
                        filled=filled,
                        rounded=rounded,
                        fontsize=10,
                        max_depth=max_depth_viz,
                        proportion=True
                    )
                    
                    plt.tight_layout()
                    st.pyplot(plt.gcf())
                    
                    # Tree interpretation guide
                    st.info("""
                    **How to Read the Decision Tree**:
                    - Each node shows a condition on a feature
                    - Samples go left (True) or right (False) based on the condition
                    - Leaf nodes show the predicted class and the proportion of samples
                    - The color of filled nodes indicates the majority class
                    - The darker the color, the higher the purity of the node
                    """)
                    
                    # Text representation of the tree
                    st.subheader("Tree as Text")
                    with st.expander("Show Text Representation"):
                        st.text(tree_viz['text'])
                    
                    # Download full tree visualization (SVG format)
                    from io import BytesIO
                    
                    # Generate high-resolution tree for download
                    plt.figure(figsize=(20, 15))
                    _ = plot_tree(
                        model,
                        feature_names=feature_names_after_preprocessing,
                        class_names=[str(c) for c in classes],
                        filled=True,
                        rounded=True,
                        fontsize=10
                    )
                    plt.tight_layout()
                    
                    # Save plot to SVG
                    buf = BytesIO()
                    plt.savefig(buf, format='svg')
                    buf.seek(0)
                    svg = buf.read().decode('utf-8')
                    
                    # Provide download link
                    st.download_button(
                        label="Download Full Tree (SVG)",
                        data=svg,
                        file_name="decision_tree.svg",
                        mime="image/svg+xml"
                    )
            
            with eval_tab3:
                st.subheader("Feature Importance Analysis")
                
                # Get feature importances
                importances = model.feature_importances_
                
                # Map importances to feature names
                feature_names_after_preprocessing = None
                preprocessor = pipeline.named_steps.get('preprocessor')
                if preprocessor and hasattr(preprocessor, 'get_feature_names_out'):
                    try:
                        feature_names_after_preprocessing = preprocessor.get_feature_names_out()
                    except:
                        pass
                
                if feature_names_after_preprocessing is not None:
                    feature_importance_df = pd.DataFrame({
                        'Feature': feature_names_after_preprocessing,
                        'Importance': importances
                    })
                else:
                    # Fallback to original feature names
                    feature_importance_df = pd.DataFrame({
                        'Feature': [f"Feature {i}" for i in range(len(importances))],
                        'Importance': importances
                    })
                
                # Sort by importance
                feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
                
                # Select visualization type
                viz_type = st.radio(
                    "Visualization Type:",
                    options=["Bar Chart", "Horizontal Bar", "Pie Chart", "Tree Map"],
                    horizontal=True
                )
                
                # Number of top features to display
                n_features = st.slider(
                    "Number of top features to display:",
                    min_value=1,
                    max_value=len(feature_importance_df),
                    value=min(10, len(feature_importance_df))
                )
                
                # Filter top N features
                top_features = feature_importance_df.head(n_features)
                
                # Create visualization based on selected type
                if viz_type == "Bar Chart":
                    fig = px.bar(
                        top_features,
                        x='Feature',
                        y='Importance',
                        title=f"Top {n_features} Feature Importances",
                        color='Importance',
                        color_continuous_scale='Blues'
                    )
                    
                elif viz_type == "Horizontal Bar":
                    fig = px.bar(
                        top_features,
                        y='Feature',
                        x='Importance',
                        title=f"Top {n_features} Feature Importances",
                        orientation='h',
                        color='Importance',
                        color_continuous_scale='Blues'
                    )
                    
                elif viz_type == "Pie Chart":
                    fig = px.pie(
                        top_features,
                        names='Feature',
                        values='Importance',
                        title=f"Top {n_features} Feature Importances Distribution"
                    )
                    
                else:  # Tree Map
                    fig = px.treemap(
                        top_features,
                        path=['Feature'],
                        values='Importance',
                        title=f"Top {n_features} Feature Importances Tree Map"
                    )
                
                # Display the visualization
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance table
                st.subheader("Feature Importance Table")
                
                # Format importance as percentage
                top_features_formatted = top_features.copy()
                top_features_formatted['Importance (%)'] = top_features_formatted['Importance'] * 100
                
                # Style the dataframe
                st.dataframe(
                    top_features_formatted.style.format({
                        'Importance': '{:.4f}',
                        'Importance (%)': '{:.2f}%'
                    }),
                    use_container_width=True
                )
                
                # Feature importance interpretation
                st.info("""
                **Understanding Feature Importance**:
                - Feature importance in Decision Trees indicates how much each feature contributes to the prediction
                - Higher values mean the feature has more influence on the decision-making process
                - It's calculated based on how much each feature decreases impurity across all nodes
                - Features with zero importance can be considered for removal
                """)
                
                # Cumulative importance
                st.subheader("Cumulative Feature Importance")
                
                # Calculate cumulative importance
                feature_importance_df['Cumulative Importance'] = feature_importance_df['Importance'].cumsum()
                
                # Create cumulative importance plot
                fig = px.line(
                    feature_importance_df,
                    x=range(1, len(feature_importance_df) + 1),
                    y='Cumulative Importance',
                    title="Cumulative Feature Importance",
                    labels={'x': 'Number of Features', 'y': 'Cumulative Importance'}
                )
                
                # Add threshold line at 95%
                fig.add_hline(y=0.95, line_dash="dash", line_color="red", annotation_text="95% Importance", annotation_position="bottom right")
                
                # Highlight number of features needed for 95% importance
                features_for_95 = (feature_importance_df['Cumulative Importance'] >= 0.95).idxmax() + 1
                fig.add_vline(x=features_for_95, line_dash="dash", line_color="green", 
                               annotation_text=f"{features_for_95} Features", annotation_position="top left")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Importance threshold for feature selection
                st.subheader("Feature Selection by Importance")
                
                importance_threshold = st.slider(
                    "Select minimum importance threshold (%):",
                    min_value=0.0,
                    max_value=float(feature_importance_df['Importance'].max() * 100),
                    value=1.0,
                    step=0.1,
                    format="%.1f%%"
                )
                
                # Get features above threshold
                important_features = feature_importance_df[feature_importance_df['Importance'] >= importance_threshold / 100]
                
                # Display selected features
                st.write(f"**{len(important_features)} features selected** (out of {len(feature_importance_df)})")
                st.write(important_features)
                
                # Option to create dataset with only important features
                if len(important_features) > 0 and len(important_features) < len(feature_importance_df):
                    if st.button("Create Dataset with Selected Features Only"):
                        # Create new dataframe with only important features
                        important_feature_names = important_features['Feature'].tolist()
                        
                        # Map preprocessed feature names back to original features
                        # This is complex and depends on the preprocessing pipeline
                        # For now, we'll give a simpler solution
                        st.info("Feature names in the importance table are from after preprocessing. The dataset would contain the corresponding original features.")
                        
                        # Store this information in session state for future feature
                        st.session_state['dt_important_features'] = important_feature_names
                        
                        st.success(f"✅ Selected {len(important_features)} important features!")
            
            with eval_tab4:
                st.subheader("Error Analysis")
                
                # Get predictions
                y_pred = pipeline.predict(X_test)
                
                # Create DataFrame with true and predicted labels
                error_df = pd.DataFrame({
                    'True Label': classes[y_test],
                    'Predicted Label': classes[y_pred],
                    'Correct': y_test == y_pred
                })
                
                # Add original features for analysis
                if isinstance(X_test, pd.DataFrame):
                    for col in X_test.columns:
                        error_df[col] = X_test[col].values
                
                # Summary of errors
                st.write(f"**Test Set Size**: {len(error_df)} samples")
                st.write(f"**Correctly Classified**: {error_df['Correct'].sum()} samples ({error_df['Correct'].mean()*100:.2f}%)")
                st.write(f"**Misclassified**: {(~error_df['Correct']).sum()} samples ({(~error_df['Correct']).mean()*100:.2f}%)")
                
                # Show misclassified examples
                st.subheader("Misclassified Examples")
                
                misclassified = error_df[~error_df['Correct']]
                if len(misclassified) > 0:
                    st.dataframe(misclassified.head(10), use_container_width=True)
                    
                    # Analyze misclassifications by class
                    st.subheader("Misclassifications by Class")
                    
                    # Create matrix of actual vs predicted for misclassified
                    misclass_matrix = pd.crosstab(
                        misclassified['True Label'], 
                        misclassified['Predicted Label'], 
                        rownames=['True'], 
                        colnames=['Predicted']
                    )
                    
                    # Display as heatmap
                    fig = px.imshow(
                        misclass_matrix,
                        labels=dict(x="Predicted", y="True", color="Count"),
                        color_continuous_scale="Reds",
                        title="Misclassification Heatmap"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Find classes with highest error rates
                    class_error_rates = {}
                    for cls in classes:
                        class_samples = error_df[error_df['True Label'] == cls]
                        if len(class_samples) > 0:
                            error_rate = (~class_samples['Correct']).mean() * 100
                            class_error_rates[cls] = error_rate
                    
                    # Sort by error rate
                    class_error_df = pd.DataFrame({
                        'Class': list(class_error_rates.keys()),
                        'Error Rate (%)': list(class_error_rates.values()),
                        'Count': [len(error_df[error_df['True Label'] == cls]) for cls in class_error_rates.keys()]
                    }).sort_values('Error Rate (%)', ascending=False)
                    
                    # Display class error rates
                    st.subheader("Error Rates by Class")
                    
                    # Bar chart of error rates
                    fig = px.bar(
                        class_error_df,
                        x='Class',
                        y='Error Rate (%)',
                        title="Error Rate by Class",
                        text='Count',
                        color='Error Rate (%)',
                        color_continuous_scale='Reds'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Suggest potential improvements
                    st.subheader("Potential Improvements")
                    
                    high_error_classes = class_error_df[class_error_df['Error Rate (%)'] > 20]['Class'].tolist()
                    
                    if high_error_classes:
                        st.write("Based on error analysis, consider these improvements:")
                        st.write(f"- **Classes with high error rates**: {', '.join(high_error_classes)}")
                        st.write("- Collect more data for these classes")
                        st.write("- Try different hyperparameters (increased tree depth, reduced minimum samples)")
                        st.write("- Consider ensemble methods like Random Forest or Gradient Boosting")
                        st.write("- Feature engineering to better separate these classes")
                    else:
                        st.write("No classes with particularly high error rates detected. The model is performing well across all classes.")
                    
                else:
                    st.success("Amazing! No misclassified examples in the test set. The model achieved 100% accuracy.")
        
        # Note about model retraining
        st.info("To retrain the model with different parameters, go back to the 'Train Model' tab.")
        
    with tab4:
        st.header("Make Predictions")
        
        # Check if a model has been trained
        if st.session_state.get('dt_trained_model') is None:
            st.warning("⚠️ You need to train a model first. Go to the 'Train Model' tab to train a Decision Tree model.")
        else:
            # Get model and data from session state
            pipeline = st.session_state.dt_pipeline
            feature_names = st.session_state.dt_feature_names
            classes = st.session_state.dt_classes
            
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
                    if st.session_state.get('dt_X_test') is not None:
                        X_pred = st.session_state.dt_X_test
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
                        if data_option == "Use test set (if available)" and st.session_state.get('dt_y_test') is not None:
                            y_true = st.session_state.dt_y_test
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
                            "decision_tree_predictions.csv",
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
                        st.text(str(e))
            
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
                                        "decision_tree_file_predictions.csv",
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
                                    st.text(str(e))
                    
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
                
                # Instructions
                with st.expander("File Format Instructions"):
                    st.markdown("""
                    ### Required File Format
                    
                    Your file should:
                    
                    1. Be in CSV or Excel format
                    2. Contain columns for all features used in training
                    3. Have the same column names as the training data
                    
                    The required feature columns are:
                    ```
                    {features}
                    ```
                    
                    No target column is needed since we're making predictions.
                    """.format(features=', '.join(feature_names)))
            
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
                   with open('decision_tree_model.pkl', 'rb') as f:
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
                with open('decision_tree_model.pkl', 'rb') as f:
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
        
else:
    show_file_required_warning()

# Footer
create_footer() 
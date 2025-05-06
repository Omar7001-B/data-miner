import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import (
    SelectKBest, 
    f_classif, 
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    RFE
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from utils import (
    load_css, 
    create_footer, 
    show_file_required_warning, 
    display_dataset_info, 
    create_card,
    get_numeric_columns,
    get_categorical_columns
)

# Filter warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="Feature Selection", page_icon="🎯", layout="wide")
load_css()

st.title("Feature Selection Interface 🎯")
st.markdown("Select the most relevant features for your machine learning models using statistical methods.")

df = st.session_state.get('df', None)
if df is not None:
    # Check if dataset is too small
    if len(df) < 10:
        st.error("Dataset is too small for reliable feature selection. Please use a larger dataset.")
        st.stop()
        
    # Display dataset metrics
    display_dataset_info()
    
    # Get numeric and categorical columns
    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_categorical_columns(df)
    
    # Check if there are enough features
    if len(df.columns) < 3:
        st.warning("This dataset has very few features. Feature selection may not be necessary.")
    
    # Create tabs for different aspects of feature selection
    tab1, tab2, tab3 = st.tabs(["Overview", "Select Features", "Feature Importance"])

    with tab1:
        st.subheader("Feature Selection Overview")
        
        # Display metrics about columns
        metrics_cols = st.columns(3)
        with metrics_cols[0]:
            st.metric("Total Features", f"{df.shape[1]}")
        with metrics_cols[1]:
            st.metric("Numeric Features", f"{len(numeric_columns)}")
        with metrics_cols[2]:
            st.metric("Categorical Features", f"{len(categorical_columns)}")
        
        # Explanation about feature selection
        st.markdown("""
        ### What is Feature Selection?
        
        Feature selection is the process of selecting a subset of relevant features for use in model construction.
        
        **Why is it important?**
        - Reduces overfitting by removing irrelevant features
        - Improves model accuracy
        - Reduces training time
        - Makes models more interpretable
        
        ### Methods Available in This Tool
        
        1. **Filter Methods** - Use statistical measures to score feature relevance:
           - ANOVA F-Test
           - Mutual Information
           - Correlation Analysis
           
        2. **Wrapper Methods** - Use model performance to evaluate feature subsets:
           - Recursive Feature Elimination (RFE)
           
        3. **Embedded Methods** - Feature importance from model training:
           - Random Forest Feature Importance
        """)
        
        # Display correlation heatmap for numeric features
        if len(numeric_columns) > 1:
            st.subheader("Feature Correlation Analysis")
            
            try:
                # Create correlation matrix
                corr_matrix = df[numeric_columns].corr()
                
                # Plot heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                cmap = sns.diverging_palette(230, 20, as_cmap=True)
                
                sns.heatmap(
                    corr_matrix, 
                    mask=mask,
                    cmap=cmap,
                    vmax=1, 
                    vmin=-1,
                    center=0,
                    square=True, 
                    linewidths=0.5, 
                    annot=True,
                    fmt=".2f",
                    ax=ax
                )
                ax.set_title("Feature Correlation Heatmap")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Highly correlated features
                st.markdown("#### Highly Correlated Features")
                st.markdown("Features with high correlation (>0.7 or <-0.7) might contain redundant information:")
                
                # Find highly correlated pairs
                highly_correlated = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            highly_correlated.append({
                                'Feature 1': corr_matrix.columns[i],
                                'Feature 2': corr_matrix.columns[j],
                                'Correlation': corr_matrix.iloc[i, j]
                            })
                
                if highly_correlated:
                    highly_corr_df = pd.DataFrame(highly_correlated)
                    st.dataframe(highly_corr_df.sort_values(by='Correlation', key=abs, ascending=False), use_container_width=True)
                    
                    st.info("Consider removing one feature from each highly correlated pair to reduce dimensionality.")
                else:
                    st.success("No highly correlated feature pairs found. Your features appear to be independent.")
            except Exception as e:
                st.error(f"Error creating correlation analysis: {str(e)}")
                st.info("This can happen with non-numeric data or if there are computation issues with your dataset.")
        else:
            st.info("You need at least two numeric columns to perform correlation analysis.")
        
        # Check for high cardinality categorical features
        if categorical_columns:
            high_cardinality = []
            for col in categorical_columns:
                unique_values = df[col].nunique()
                if unique_values > 10:
                    high_cardinality.append({
                        'Feature': col,
                        'Unique Values': unique_values,
                        'Unique %': (unique_values / len(df)) * 100
                    })
            
            if high_cardinality:
                st.subheader("High Cardinality Categorical Features")
                st.markdown("These categorical features have many unique values and might need special handling:")
                st.dataframe(pd.DataFrame(high_cardinality), use_container_width=True)
                st.info("High cardinality features might need to be encoded or transformed before feature selection.")

    with tab2:
        st.subheader("Select Features")
        
        # Initialize session state for target column
        if 'target_column' not in st.session_state:
            st.session_state.target_column = None
        
        # Target column selection
        st.markdown("### Step 1: Select Target Column")
        
        target_column = st.selectbox(
            "Select your target variable (what you want to predict)",
            options=df.columns.tolist(),
            index=None if st.session_state.target_column is None else df.columns.tolist().index(st.session_state.target_column)
        )
        
        if target_column:
            st.session_state.target_column = target_column
            
            # Check if target column has enough non-null values
            if df[target_column].isnull().sum() > 0.3 * len(df):
                st.warning(f"Warning: The target column '{target_column}' has more than 30% missing values, which may affect results.")
            
            # Determine if classification or regression based on target type
            is_categorical = target_column in categorical_columns or df[target_column].nunique() < 10
            problem_type = "Classification" if is_categorical else "Regression"
            
            if is_categorical:
                unique_classes = df[target_column].nunique()
                if unique_classes < 2:
                    st.error("Target column must have at least 2 unique classes for classification problems.")
                    st.stop()
                elif unique_classes > 10:
                    st.warning(f"Target column has {unique_classes} unique classes. Consider simplifying for better results.")
            
            st.success(f"Target column: {target_column} (Detected as {problem_type} problem)")
            
            # Feature selection method
            st.markdown("### Step 2: Choose Feature Selection Method")
            
            method = st.radio(
                "Select feature selection method",
                ["Statistical Tests", "Recursive Feature Elimination (RFE)", "Random Forest Importance"],
                help="Different methods use different approaches to rank features"
            )
            
            # Feature columns selection
            st.markdown("### Step 3: Select Candidate Features")
            
            # Initialize feature_columns in session state if not exists
            if 'feature_columns' not in st.session_state:
                st.session_state.feature_columns = []
            
            # Get all columns except target
            all_cols = [col for col in df.columns if col != target_column]
            
            if len(all_cols) < 2:
                st.error("Not enough feature columns for selection. You need at least 2 potential features.")
                st.stop()
            
            # Allow users to select either all columns or specific ones
            use_all_cols = st.checkbox("Use all available columns", value=True)
            
            if use_all_cols:
                feature_columns = all_cols
            else:
                feature_columns = st.multiselect(
                    "Select features to evaluate",
                    options=all_cols,
                    default=st.session_state.feature_columns if st.session_state.feature_columns else []
                )
                
                if not feature_columns:
                    st.warning("Please select at least one feature column.")
            
            st.session_state.feature_columns = feature_columns
            
            # Additional parameters based on method
            st.markdown("### Step 4: Configure Method Parameters")
            
            param_col1, param_col2 = st.columns(2)
            
            with param_col1:
                if method == "Statistical Tests":
                    if is_categorical:  # Classification
                        stat_test = st.selectbox(
                            "Statistical test",
                            ["ANOVA F-test", "Mutual Information"],
                            help="F-test measures linear dependency, mutual information captures any dependency"
                        )
                    else:  # Regression
                        stat_test = st.selectbox(
                            "Statistical test",
                            ["F-test", "Mutual Information"],
                            help="F-test measures linear dependency, mutual information captures any dependency"
                        )
                
                elif method == "Recursive Feature Elimination (RFE)":
                    if is_categorical:  # Classification
                        model_type = st.selectbox(
                            "Base estimator",
                            ["Logistic Regression", "Random Forest"],
                            help="The model used for recursive feature elimination"
                        )
                    else:  # Regression
                        model_type = st.selectbox(
                            "Base estimator",
                            ["Linear Regression", "Random Forest"],
                            help="The model used for recursive feature elimination"
                        )
                
                elif method == "Random Forest Importance":
                    n_estimators = st.slider(
                        "Number of trees",
                        min_value=10,
                        max_value=200,
                        value=100,
                        step=10,
                        help="More trees provide more stable results but take longer"
                    )
            
            with param_col2:
                # Number of features to select
                max_features = min(20, len(feature_columns))
                k_features = st.slider(
                    "Number of top features to select",
                    min_value=1,
                    max_value=max_features,
                    value=min(5, max_features),
                    help="Select how many top features to keep"
                )
                
                # Handle categorical features
                if any(col in categorical_columns for col in feature_columns):
                    handle_categorical = st.selectbox(
                        "How to handle categorical features?",
                        ["One-hot encode", "Skip categorical features"],
                        help="Categorical features need encoding before feature selection"
                    )
                else:
                    handle_categorical = "Skip categorical features"
            
            # Apply feature selection
            if st.button("Run Feature Selection", disabled=not feature_columns):
                if not feature_columns:
                    st.error("Please select at least one feature column.")
                else:
                    with st.spinner("Running feature selection..."):
                        try:
                            # Prepare the data
                            X = df[feature_columns].copy()
                            y = df[target_column].copy()
                            
                            # Handle missing values first
                            if X.isnull().any().any() or y.isnull().any():
                                st.warning("Missing values detected. Dropping rows with missing values.")
                                non_null_idx = X.dropna().index.intersection(y.dropna().index)
                                X = X.loc[non_null_idx]
                                y = y.loc[non_null_idx]
                                
                                if len(X) < 10:
                                    st.error("After removing missing values, there are too few rows left for analysis.")
                                    st.stop()
                            
                            # Handle categorical features
                            categorical_in_features = [col for col in feature_columns if col in categorical_columns]
                            if categorical_in_features and handle_categorical == "One-hot encode":
                                try:
                                    X = pd.get_dummies(X, columns=categorical_in_features, drop_first=True)
                                    if X.empty:
                                        st.error("One-hot encoding resulted in an empty dataframe. Try using a different approach.")
                                        st.stop()
                                except Exception as e:
                                    st.error(f"Error in one-hot encoding: {str(e)}")
                                    st.stop()
                            elif categorical_in_features and handle_categorical == "Skip categorical features":
                                numeric_only = [col for col in feature_columns if col not in categorical_columns]
                                if not numeric_only:
                                    st.error("No numeric features selected. Please select numeric features or use one-hot encoding.")
                                    st.stop()
                                X = X[numeric_only]
                            
                            # Check if we have enough features after preprocessing
                            if X.shape[1] < 2:
                                st.error("Not enough features remaining after preprocessing. Need at least 2 features.")
                                st.stop()
                            
                            # Apply feature selection based on method
                            feature_names = X.columns.tolist()
                            
                            if method == "Statistical Tests":
                                try:
                                    if is_categorical:  # Classification
                                        if stat_test == "ANOVA F-test":
                                            selector = SelectKBest(f_classif, k=min(k_features, len(feature_names)))
                                        else:  # Mutual Information
                                            selector = SelectKBest(
                                                lambda X, y: mutual_info_classif(
                                                    X, y, random_state=42
                                                ), 
                                                k=min(k_features, len(feature_names))
                                            )
                                    else:  # Regression
                                        if stat_test == "F-test":
                                            selector = SelectKBest(f_regression, k=min(k_features, len(feature_names)))
                                        else:  # Mutual Information
                                            selector = SelectKBest(
                                                lambda X, y: mutual_info_regression(
                                                    X, y, random_state=42
                                                ), 
                                                k=min(k_features, len(feature_names))
                                            )
                                    
                                    X_new = selector.fit_transform(X, y)
                                    scores = selector.scores_
                                    
                                    # Create a DataFrame of features and their scores
                                    feature_scores = pd.DataFrame({
                                        'Feature': feature_names,
                                        'Score': scores
                                    })
                                    feature_scores = feature_scores.sort_values('Score', ascending=False)
                                    
                                    # Get selected features
                                    mask = selector.get_support()
                                    selected_features = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
                                except Exception as e:
                                    st.error(f"Error in Statistical Tests: {str(e)}")
                                    st.info("Try another method or check your data for issues.")
                                    st.stop()
                                
                            elif method == "Recursive Feature Elimination (RFE)":
                                try:
                                    if is_categorical:  # Classification
                                        if model_type == "Logistic Regression":
                                            # For multiclass problems, use multinomial
                                            if df[target_column].nunique() > 2:
                                                estimator = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
                                            else:
                                                estimator = LogisticRegression(max_iter=1000)
                                        else:  # Random Forest
                                            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
                                    else:  # Regression
                                        if model_type == "Linear Regression":
                                            estimator = LinearRegression()
                                        else:  # Random Forest
                                            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
                                    
                                    # Adjust step size based on feature count to avoid issues with small feature sets
                                    step_size = max(1, int(X.shape[1] * 0.1))  # Use 10% or at least 1
                                    selector = RFE(estimator, n_features_to_select=min(k_features, len(feature_names)), step=step_size)
                                    selector = selector.fit(X, y)
                                    
                                    # Get rankings and selected features
                                    rankings = selector.ranking_
                                    feature_scores = pd.DataFrame({
                                        'Feature': feature_names,
                                        'Ranking': rankings
                                    })
                                    feature_scores = feature_scores.sort_values('Ranking')
                                    
                                    # Get selected features
                                    selected_features = [feature_names[i] for i in range(len(feature_names)) if selector.support_[i]]
                                except Exception as e:
                                    st.error(f"Error in Recursive Feature Elimination: {str(e)}")
                                    st.info("Try another method or check your data for issues.")
                                    st.stop()
                                
                            else:  # Random Forest Importance
                                try:
                                    if is_categorical:  # Classification
                                        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                                    else:  # Regression
                                        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                                        
                                    model.fit(X, y)
                                    importances = model.feature_importances_
                                    
                                    # Create a DataFrame of features and their importance
                                    feature_scores = pd.DataFrame({
                                        'Feature': feature_names,
                                        'Importance': importances
                                    })
                                    feature_scores = feature_scores.sort_values('Importance', ascending=False)
                                    
                                    # Get selected features (top k)
                                    selected_features = feature_scores['Feature'].tolist()[:min(k_features, len(feature_names))]
                                except Exception as e:
                                    st.error(f"Error in Random Forest Importance: {str(e)}")
                                    st.info("Try another method or check your data for issues.")
                                    st.stop()
                            
                            # Store results in session state
                            st.session_state['feature_selection_results'] = {
                                'selected_features': selected_features,
                                'feature_scores': feature_scores,
                                'method': method,
                                'problem_type': problem_type
                            }
                            
                            # Show success message
                            st.success(f"✅ Feature selection complete! Selected {len(selected_features)} top features.")
                        except Exception as e:
                            st.error(f"An unexpected error occurred: {str(e)}")
                            st.info("Please check your data and try again with different parameters.")
            
            # Display results if available
            if 'feature_selection_results' in st.session_state:
                results = st.session_state['feature_selection_results']
                
                st.subheader("Feature Selection Results")
                
                # Show selected features
                st.markdown("#### Selected Features")
                
                # Ensure we don't create more columns than features
                num_features = len(results['selected_features'])
                if num_features > 0:
                    # Limit to maximum 6 columns to prevent layout issues
                    cols_per_row = min(6, num_features)
                    num_rows = (num_features + cols_per_row - 1) // cols_per_row  # Ceiling division
                    
                    for row in range(num_rows):
                        selected_cols = st.columns(cols_per_row)
                        for i in range(cols_per_row):
                            idx = row * cols_per_row + i
                            if idx < num_features:
                                with selected_cols[i]:
                                    st.metric(f"Rank {idx+1}", results['selected_features'][idx])
                else:
                    st.warning("No features were selected. Try adjusting your parameters.")
                
                # Show feature scores
                st.markdown("#### Feature Ranking")
                
                # Customize display based on method
                if results['method'] == "Statistical Tests":
                    score_col = "Score"
                    chart_title = "Feature Scores"
                elif results['method'] == "Recursive Feature Elimination (RFE)":
                    score_col = "Ranking"
                    chart_title = "Feature Rankings (lower is better)"
                else:  # Random Forest Importance
                    score_col = "Importance"
                    chart_title = "Feature Importance"
                
                # Display feature scores in a table
                st.dataframe(results['feature_scores'], use_container_width=True)
                
                # Visualize feature scores
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if results['method'] == "Recursive Feature Elimination (RFE)":
                        # For RFE, lower rank is better
                        plot_data = results['feature_scores'].head(min(10, len(results['feature_scores'])))
                        sns.barplot(
                            x=score_col, 
                            y='Feature', 
                            data=plot_data,
                            ax=ax,
                            color='skyblue'
                        )
                        plt.title(chart_title)
                        plt.xlabel("Ranking (lower is better)")
                    else:
                        # Sort by score descending and plot top 10
                        plot_data = results['feature_scores'].sort_values(score_col, ascending=False).head(min(10, len(results['feature_scores'])))
                        sns.barplot(
                            x=score_col, 
                            y='Feature', 
                            data=plot_data,
                            ax=ax,
                            color='skyblue'
                        )
                        plt.title(chart_title)
                        plt.xlabel(score_col)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error creating visualization: {str(e)}")
                
                # Create dataframe with only selected features
                if results['selected_features'] and st.button("Create Dataframe with Selected Features"):
                    try:
                        selected_df = df[[target_column] + results['selected_features']].copy()
                        st.session_state['selected_features_df'] = selected_df
                        st.success("Created dataframe with only selected features!")
                        
                        # Show download button
                        st.download_button(
                            "📥 Download Selected Features Dataset as CSV", 
                            selected_df.to_csv(index=False), 
                            file_name="selected_features_data.csv",
                            mime="text/csv"
                        )
                        
                        # Offer option to replace original dataframe
                        if st.button("📥 Replace Original Data with Selected Features"):
                            st.session_state['df'] = selected_df.copy()
                            st.success("Original data replaced with selected features dataset!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error creating dataframe: {str(e)}")
                
                # Show preview if created
                if 'selected_features_df' in st.session_state:
                    with st.expander("Preview Dataset with Selected Features", expanded=True):
                        st.dataframe(st.session_state['selected_features_df'].head(10), use_container_width=True)

    with tab3:
        st.subheader("Feature Importance Visualization")
        
        # Show feature importance visualization if results are available
        if 'feature_selection_results' in st.session_state:
            results = st.session_state['feature_selection_results']
            
            # Plot different visualization types
            visual_type = st.radio(
                "Choose visualization type",
                ["Bar Chart", "Radar Chart", "Tree Map"],
                help="Different ways to visualize feature importance"
            )
            
            # Get feature scores
            feature_scores = results['feature_scores']
            
            # Check if we have enough features for visualization
            if len(feature_scores) < 2:
                st.warning("Not enough features for advanced visualization. A bar chart will be shown.")
                visual_type = "Bar Chart"
            
            # Customize based on method
            if results['method'] == "Statistical Tests":
                score_col = "Score"
                title = "Feature Score Comparison"
            elif results['method'] == "Recursive Feature Elimination (RFE)":
                score_col = "Ranking"
                title = "Feature Ranking Comparison (lower is better)"
                # Convert rankings to inverse scores for better visualization
                try:
                    max_rank = feature_scores['Ranking'].max()
                    feature_scores['InverseRank'] = max_rank - feature_scores['Ranking'] + 1
                    score_col = "InverseRank"
                    title = "Feature Ranking Comparison (higher is better)"
                except Exception:
                    st.warning("Could not create inverse rankings. Using original rankings.")
                    score_col = "Ranking"
            else:  # Random Forest Importance
                score_col = "Importance"
                title = "Feature Importance Comparison"
            
            try:
                if visual_type == "Bar Chart":
                    # Sort and get top 15 features
                    if results['method'] == "Recursive Feature Elimination (RFE)" and 'InverseRank' in feature_scores:
                        plot_data = feature_scores.sort_values('InverseRank', ascending=False).head(min(15, len(feature_scores)))
                        score_col = 'InverseRank'
                    else:
                        plot_data = feature_scores.sort_values(score_col, ascending=False).head(min(15, len(feature_scores)))
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    bars = sns.barplot(
                        x=score_col, 
                        y='Feature', 
                        data=plot_data,
                        ax=ax,
                        palette='viridis'
                    )
                    
                    # Highlight selected features
                    selected_features = results['selected_features']
                    for i, (_, row) in enumerate(plot_data.iterrows()):
                        if row['Feature'] in selected_features:
                            if i < len(bars.patches):  # Ensure we don't go out of bounds
                                bars.patches[i].set_facecolor('orange')
                    
                    plt.title(title)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                elif visual_type == "Radar Chart" and len(feature_scores) >= 3:
                    # Use top 8 features for radar chart (otherwise too crowded)
                    if results['method'] == "Recursive Feature Elimination (RFE)" and 'InverseRank' in feature_scores:
                        plot_data = feature_scores.sort_values('InverseRank', ascending=False).head(min(8, len(feature_scores)))
                        score_col = 'InverseRank'
                    else:
                        plot_data = feature_scores.sort_values(score_col, ascending=False).head(min(8, len(feature_scores)))
                    
                    # Prepare data for radar chart
                    features = plot_data['Feature'].tolist()
                    values = plot_data[score_col].tolist()
                    
                    # Number of variables
                    N = len(features)
                    
                    # Create angles for each feature
                    angles = [n / float(N) * 2 * np.pi for n in range(N)]
                    angles += angles[:1]  # Close the loop
                    
                    # Add values for the last point to close the loop
                    values += values[:1]
                    
                    # Create plot
                    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
                    
                    # Draw the outline of the chart
                    plt.xticks(angles[:-1], features, color='grey', size=12)
                    
                    # Plot data
                    ax.plot(angles, values, linewidth=1, linestyle='solid')
                    
                    # Fill area
                    ax.fill(angles, values, 'skyblue', alpha=0.5)
                    
                    plt.title(title, size=15, color='black', y=1.1)
                    st.pyplot(fig)
                    
                else:  # Tree Map or fallback if Radar Chart has too few points
                    if visual_type == "Radar Chart" and len(feature_scores) < 3:
                        st.warning("Not enough features for a radar chart. Showing tree map instead.")
                    
                    # Try to import squarify, show alternative if not available
                    try:
                        import squarify
                        has_squarify = True
                    except ImportError:
                        has_squarify = False
                        st.warning("The 'squarify' package is not installed. Showing bar chart instead.")
                        
                    if has_squarify:
                        # Prepare data for tree map
                        if results['method'] == "Recursive Feature Elimination (RFE)" and 'InverseRank' in feature_scores:
                            plot_data = feature_scores.sort_values('InverseRank', ascending=False).head(min(12, len(feature_scores)))
                            score_col = 'InverseRank'
                        else:
                            plot_data = feature_scores.sort_values(score_col, ascending=False).head(min(12, len(feature_scores)))
                        
                        features = plot_data['Feature'].tolist()
                        values = plot_data[score_col].tolist()
                        
                        # Ensure all values are positive for treemap
                        if min(values) < 0:
                            values = [v - min(values) + 0.01 for v in values]
                        elif min(values) == 0:
                            values = [v + 0.01 if v == 0 else v for v in values]
                        
                        # Create colors based on whether feature is selected
                        colors = ['orange' if f in results['selected_features'] else 'skyblue' for f in features]
                        
                        # Create tree map
                        fig, ax = plt.subplots(figsize=(12, 8))
                        squarify.plot(sizes=values, label=features, alpha=0.6, color=colors, ax=ax)
                        plt.axis('off')
                        plt.title(title)
                        st.pyplot(fig)
                    else:
                        # Fallback to bar chart if squarify not available
                        if results['method'] == "Recursive Feature Elimination (RFE)" and 'InverseRank' in feature_scores:
                            plot_data = feature_scores.sort_values('InverseRank', ascending=False).head(min(15, len(feature_scores)))
                            score_col = 'InverseRank'
                        else:
                            plot_data = feature_scores.sort_values(score_col, ascending=False).head(min(15, len(feature_scores)))
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        bars = sns.barplot(
                            x=score_col, 
                            y='Feature', 
                            data=plot_data,
                            ax=ax,
                            palette='viridis'
                        )
                        
                        # Highlight selected features
                        selected_features = results['selected_features']
                        for i, (_, row) in enumerate(plot_data.iterrows()):
                            if row['Feature'] in selected_features and i < len(bars.patches):
                                bars.patches[i].set_facecolor('orange')
                        
                        plt.title(title)
                        plt.tight_layout()
                        st.pyplot(fig)
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
                st.info("Try another visualization type or check your data.")
        else:
            st.info("Run feature selection first to see visualizations here.")
            
            # Sample visualization for demonstration
            st.markdown("#### Sample Feature Importance Visualization")
            
            # Create sample data
            sample_features = ['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E']
            sample_scores = [0.85, 0.72, 0.63, 0.51, 0.42]
            
            # Plot sample chart
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = sns.barplot(
                    x=sample_scores,
                    y=sample_features,
                    ax=ax,
                    palette='viridis'
                )
                plt.title("Sample Feature Importance (for demonstration)")
                plt.xlabel("Importance")
                st.pyplot(fig)
                
                st.caption("This is a sample visualization. Run feature selection to see your actual results.")
            except Exception:
                st.info("Sample visualization could not be created.")
else:
    show_file_required_warning()

# Show installation note for required packages
st.sidebar.markdown("### Additional Tools")
if st.sidebar.checkbox("Show Installation Instructions"):
    st.sidebar.code("""
    # Required packages for all features
    pip install scikit-learn pandas matplotlib seaborn numpy
    
    # Optional package for treemap visualizations
    pip install squarify
    """)
    st.sidebar.info("All core features will work without squarify, but treemap visualizations require it.")

# Footer
create_footer() 
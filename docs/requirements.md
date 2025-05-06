# DataMiner - Data Mining and Machine Learning Application

## Project Information
- **Project Name**: DataMiner
- **Developer**: Omar Abbas
- **GitHub Repository**: [https://github.com/Omar7001-B/data-miner](https://github.com/Omar7001-B/data-miner)
- **License**: MIT

## 1. Introduction

The goal of this project is to develop a data mining and machine learning application that enables users to inspect, clean, transform, and analyze datasets while applying machine learning models for classification or regression.

This project is implemented as a web application using Streamlit, providing an intuitive interface designed for both beginners and experienced data scientists.

## 2. Objectives

- Provide an interactive tool for handling datasets
- Enable data preprocessing through cleaning, transformation, and feature selection
- Implement classification and regression models for predictive analysis
- Offer visualization tools to explore both raw data and model results

## 3. System Features

### 3.1 Data Handling
- Allow users to upload/load CSV, Excel, and JSON files
- Display basic dataset information (column names, data types, missing values, statistics)
- Interactive data preview with pagination
- Comprehensive dataset profiling (statistics, distributions, correlations)

### 3.2 Data Cleaning
- Handle missing values (drop rows, fill with mean/median)
- Remove duplicate entries
- Convert categorical values (One-Hot Encoding or Label Encoding)
- Interactive visualization of missing data patterns

### 3.3 Data Transformation
- **Scaling and Normalization**:
  - Min-Max scaling
  - Z-score normalization
  - Decimal scaling
- Feature selection: Users choose relevant features for training

### 3.4 Machine Learning Models
#### Classification Models:
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)

#### Regression Models:
- Linear Regression

### 3.5 Model Evaluation & Visualization
- **Classification Metrics**: Accuracy, Precision, Recall, Confusion Matrix
- **Regression Metrics**: Mean Squared Error, R² Score, Residual Plots
- **Custom Data Visualizations**: Histograms, scatter plots, correlation heatmaps

### 3.6 User Interaction & Results
- Provide options to apply transformations and train models interactively
- Allow users to make predictions using the trained models

## 4. Technology Stack

### Web Application
- **Frontend**: Streamlit
- **Backend**: Python 3.9+
- **Libraries**:
  - Pandas (data manipulation)
  - NumPy (numerical operations)
  - Scikit-learn (machine learning)
  - Matplotlib & Seaborn (visualization)
  - Plotly (interactive visualizations)

## 5. Implementation Approach
1. **Phase 1**: Data handling and basic preprocessing
2. **Phase 2**: Advanced preprocessing and transformation
3. **Phase 3**: Model implementation and evaluation
4. **Phase 4**: UI refinement and user experience optimization

## 6. Assumptions & Constraints
- Users will provide well-structured CSV files
- Model training should be performed on moderate-sized datasets

## 7. Future Enhancements
- Expand model selection (Random Forest, SVM, Neural Networks)
- Implement clustering and association rule mining
- Provide dataset storage and retrieval for future use
- Enable the exporting of processed data and trained models
- Time series forecasting models
- AutoML capabilities for automated model selection

## 8. Getting Started
```bash
# Clone the repository
git clone https://github.com/Omar7001-B/data-miner.git
cd data-miner

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run Home.py
```
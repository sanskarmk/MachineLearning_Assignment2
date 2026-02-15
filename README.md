# 🔍 Machine Learning Based Phishing Website Detection System

## Machine Learning – Assignment 2  
**M.Tech (AIML ) Program, BITS Pilani**

---

## a. Problem Statement

The proliferation of online services has made the internet an integral part of daily life. However, this digital expansion has also given rise to sophisticated cyber threats, particularly phishing attacks. Phishing websites masquerade as legitimate web pages to deceive users into disclosing sensitive information such as passwords, credit card details, and personal identification data. These fraudulent sites pose significant security risks to individuals and organizations worldwide.

This project addresses the critical need for automated phishing detection systems. The objective is to develop and compare multiple machine learning classification algorithms that can accurately distinguish between legitimate and phishing websites based on numerical features extracted from URLs and web page characteristics. By analyzing patterns in URL structure, domain properties, and other technical indicators, these models can provide real-time protection against phishing threats. An interactive web application built using Streamlit demonstrates the practical deployment of these classification models.

---

## b. Dataset Description

The dataset used in this project is specifically designed for phishing website detection and contains comprehensive feature engineering from web URLs.

**Dataset Characteristics:**
- **Classification Type:** Binary Classification Problem  
- **Total Instances:** Approximately 11,000 website samples  
- **Number of Features:** 87 numerical features engineered from URLs  
- **Target Variable:** `status`  
  - Class 0: `legitimate` (safe websites)  
  - Class 1: `phishing` (malicious websites)  

**Feature Engineering:**
The dataset contains 87 numerical features derived from various URL and website properties including:
- URL length and structure metrics
- Presence of special characters (e.g., @, //, -)
- SSL certificate information and HTTPS usage
- Domain registration details and age
- Query parameter characteristics
- Subdomain count and depth
- IP address presence in URL
- Port number usage
- Redirect patterns

**Data Quality:**
The dataset is clean with no missing values. The URL text column is excluded from modeling as it's non-numeric. All 87 features are numerical and ready for machine learning algorithms. The dataset exhibits balanced class distribution, making it suitable for training robust classification models.

---

## c. Machine Learning Models Used and Evaluation Metrics

This project implements six different classification algorithms to comprehensively evaluate various machine learning approaches for phishing detection. All models were trained on identical training sets and evaluated on the same test set using an **80-20 stratified train-test split** to ensure fair and unbiased comparison.

**Offline model training (as per assignment requirement):** All 6 models are trained offline and saved as `.pkl` files under `model/` (and committed to GitHub). The Streamlit app loads these at startup.

### Classification Algorithms Implemented

1. **Logistic Regression** - Linear probabilistic classifier using sigmoid function
2. **Decision Tree Classifier** - Hierarchical rule-based tree structure
3. **K-Nearest Neighbors (KNN)** - Instance-based lazy learning algorithm
4. **Naive Bayes (Gaussian)** - Probabilistic classifier based on Bayes' theorem
5. **Random Forest** - Ensemble of multiple decision trees with voting
6. **XGBoost** - Gradient boosting ensemble with regularization

---

### Evaluation Metrics Calculated

For comprehensive model assessment, six evaluation metrics were computed for each classifier:

- **Accuracy**: Overall proportion of correct predictions
- **AUC (Area Under ROC Curve)**: Model's discrimination ability between classes
- **Precision**: Correctness of positive predictions (phishing detection accuracy)
- **Recall (Sensitivity)**: Ability to identify all phishing instances
- **F1 Score**: Harmonic balance between precision and recall
- **MCC (Matthews Correlation Coefficient)**: Balanced measure considering all confusion matrix elements

---

### Comprehensive Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.9361 | 0.9814 | 0.9392 | 0.9326 | 0.9359 | 0.8723 |
| Decision Tree | 0.9339 | 0.9339 | 0.9291 | 0.9396 | 0.9343 | 0.8679 |
| KNN | 0.9396 | 0.9788 | 0.9539 | 0.9239 | 0.9387 | 0.8797 |
| Naive Bayes | 0.7428 | 0.8051 | 0.6920 | 0.8749 | 0.7728 | 0.5035 |
| Random Forest | 0.9611 | 0.9935 | 0.9575 | 0.9650 | 0.9612 | 0.9222 |
| XGBoost | 0.9689 | 0.9948 | 0.9645 | 0.9738 | 0.9691 | 0.9379 |

---

### Detailed Model Analysis and Observations

| ML Model | Key Observations and Performance Characteristics |
|---------|--------------------------------------------------|
| Logistic Regression | Demonstrates solid baseline performance with good precision but relatively lower recall. The linear decision boundary limits its ability to capture complex non-linear patterns in phishing features. Computationally efficient and provides interpretable coefficients for feature importance. |
| Decision Tree | Captures non-linear relationships effectively through hierarchical splits. Shows strong performance but may exhibit slight overfitting tendencies on training data. Tree visualization provides excellent interpretability of decision logic. No feature scaling required. |
| KNN | Achieves balanced performance by utilizing local neighborhood information. Computationally expensive during prediction phase as it requires distance calculations to all training samples. Sensitive to feature scaling and curse of dimensionality with 87 features. |
| Naive Bayes | Fastest training and prediction times due to probabilistic independence assumptions. Performance limited by the assumption that features are independent, which may not hold true for URL-based features. Works well despite feature correlations and provides probabilistic interpretations. |
| Random Forest (Ensemble) | Delivers excellent performance through ensemble averaging of multiple trees. Reduces overfitting compared to single decision tree. Provides feature importance rankings and handles non-linear interactions well. Robust to outliers and doesn't require feature scaling. |
| XGBoost (Ensemble) | Achieves the highest overall performance across all metrics. Gradient boosting sequentially corrects errors from previous trees. Built-in regularization prevents overfitting. Handles complex feature interactions and provides the most accurate phishing detection. Best choice for deployment. |

---

## Interactive Streamlit Web Application

An interactive web-based application was developed using Streamlit framework to demonstrate the trained machine learning models in action. The application provides a user-friendly interface for model evaluation and comparison.

### Application Features:

- **CSV Upload Functionality**: Users can upload custom test datasets in CSV format for evaluation
- **Model Selection**: Dropdown menu to choose from all six implemented machine learning algorithms
- **Comprehensive Metrics Display**: Visual presentation of all six evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- **Confusion Matrix Visualization**: Tabular display showing true positives, false positives, true negatives, and false negatives
- **Classification Report**: Detailed per-class performance metrics including support values
- **Responsive Design**: Clean, organized layout with intuitive navigation

---

## Project Repository Structure

```
MachineLearning_Assignment2/
│
├── app.py                          # Streamlit web application (loads pre-trained models)
├── train_all_models_offline.py     # Offline training script (generates PKLs + test_data.csv)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── test_data.csv                   # Test data split (2,286 rows) for Streamlit upload
├── model_comparison_results.csv    # Pre-computed comparison metrics
│
└── model/                          # Pre-trained model PKL files (committed to GitHub)
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    └── scaler.pkl                  # StandardScaler for models that need scaling
```

---

## Local Execution Instructions

To run this project on your local machine:

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MachineLearning
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Streamlit application**
   ```bash
   streamlit run app.py
   ```

5. **Access the application**
   - The app will open automatically in your default browser
   - Or navigate to: `http://localhost:8501`

---

## Cloud Deployment

The application is deployed on **Streamlit Community Cloud** for public access. This cloud-based deployment allows anyone to interact with the models without local installation. The live application link is included in the assignment submission PDF.

**Deployment Process:**
- GitHub repository connected to Streamlit Cloud
- Automatic deployment on code commits
- Free hosting with public access
- requirements.txt ensures dependency compatibility

---

## Conclusion and Key Insights

This project successfully demonstrates a complete end-to-end machine learning workflow for phishing website detection:

1. **Data Preprocessing**: Proper handling of numerical features, label encoding, and stratified splitting
2. **Model Implementation**: Six diverse algorithms covering linear, tree-based, instance-based, probabilistic, and ensemble methods
3. **Comprehensive Evaluation**: Multiple metrics provide holistic performance assessment
4. **Practical Deployment**: Interactive web application makes models accessible to end-users

**Key Findings:**
- Ensemble methods (Random Forest and XGBoost) significantly outperform simpler classifiers
- XGBoost achieves the best performance with 96.89% accuracy and 0.9379 MCC score
- Naive Bayes shows the lowest performance (74.28% accuracy) due to feature independence assumptions not holding well for URL-derived features
- Feature scaling impacts KNN and Logistic Regression but not tree-based models
- Most models demonstrate strong AUC scores (>0.93), indicating good class discrimination
- A `test_data.csv` file is provided in the repository for quick testing via the Streamlit app

**Practical Impact:**
The high performance of ensemble models makes them suitable for real-world phishing detection systems. The deployed web application demonstrates how machine learning can be integrated into cybersecurity tools for automated threat detection.

---

## Author Information

**Name:** Sanskar Maheshkumar Khandelwal  
**Student ID:** 2025AA05332  
**Email:** 2025AA05332@wilp.bits-pilani.ac.in  
**Program:** M.Tech (AIML), BITS Pilani  
**Institution:** BITS Pilani - Work Integrated Learning Programmes Division  
**Course:** Machine Learning  
**Assignment:** Assignment 2 - Classification Models and Deployment

---

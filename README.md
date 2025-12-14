# Heart Disease Prediction

A comprehensive machine learning project for predicting heart disease risk using advanced feature selection techniques and ensemble methods.

## Overview

This project develops and compares multiple machine learning models to predict the likelihood of heart disease in patients. The analysis includes exploratory data analysis, statistical testing, hyperparameter optimization, and advanced feature selection methods to identify the most influential predictors of heart disease.

## Key Features

- **Comprehensive EDA**: Statistical analysis with t-tests and chi-squared tests to identify significant features
- **Multiple ML Models**: Logistic Regression, KNN, SVM, Decision Tree, Random Forest, MLP, LightGBM, and Stacking Classifier
- **Hyperparameter Optimization**: Grid Search with 5-fold Cross-Validation for all models
- **Feature Selection**: Comparison of Particle Swarm Optimization (PSO) and Minimum Redundancy Maximum Relevance (mRMR) methods
- **Model Interpretability**: Feature importance analysis using LightGBM gain metrics and SHAP (Shapley Additive Explanations) values
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, and ROC-AUC

## Project Structure

```
├── EDA.ipynb                    # Exploratory data analysis and visualization
├── models.ipynb                 # Model training and evaluation
├── feature_selection.ipynb      # PSO and mRMR feature selection comparison
├── heart.csv                    # Dataset
└── README.md
```

## Dataset

**Source**: [Heart Disease Dataset - Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

**Size**: 1,025 patients with 14 features

**Features**:
- `age`: Age in years
- `sex`: Gender (1 = male, 0 = female)
- `cp`: Chest pain type (0-3)
- `trestbps`: Resting blood pressure (mm Hg)
- `chol`: Serum cholesterol (mg/dl)
- `fbs`: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- `restecg`: Resting electrocardiographic results (0-2)
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise induced angina (1 = yes, 0 = no)
- `oldpeak`: ST depression induced by exercise
- `slope`: Slope of peak exercise ST segment (0-2)
- `ca`: Number of major vessels colored by fluoroscopy (0-4)
- `thal`: Thalassemia (0-3)
- `target`: Heart disease diagnosis (1 = disease, 0 = no disease)

## Methodology

### 1. Exploratory Data Analysis ([EDA.ipynb](EDA.ipynb))
- Univariate and bivariate analysis with visualizations
- Statistical hypothesis testing (t-test for continuous, chi-squared for categorical features)
- Identified that all features except `fbs` are statistically significant predictors

### 2. Model Training ([models.ipynb](models.ipynb))
Trained and optimized the following models using GridSearchCV:

| Model | Best Test F1-Score | Test AUC |
|-------|-------------------|----------|
| Logistic Regression | 0.849 | 0.929 |
| K-Nearest Neighbors | 0.851 | 0.956 |
| Support Vector Machine | 0.844 | 0.920 |
| Decision Tree | 0.877 | 0.941 |
| Random Forest | 0.990 | 0.999 |
| Multi-Layer Perceptron | 0.971 | 0.995 |
| Stacking Classifier | 0.908 | 0.978 |
| **LightGBM** | **0.981** | **0.995** |

**Best Performing Model**: LightGBM with 98.1% F1-Score and 99.5% AUC

### 3. Feature Selection ([feature_selection.ipynb](feature_selection.ipynb))

Compared two feature selection methods:

**Particle Swarm Optimization (PSO)**:
- Selected 11 features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, oldpeak, ca, thal
- LightGBM Test F1-Score: 0.975, AUC: 0.997

**Minimum Redundancy Maximum Relevance (mRMR)**:
- Selected 11 features: chol, exang, thal, ca, cp, slope, sex, oldpeak, thalach, restecg, trestbps
- LightGBM Test F1-Score: 0.971, AUC: 0.993

**Top 5 Most Important Features** (from feature importance analysis):
1. Chest pain type (cp)
2. Thalassemia (thal)
3. Age
4. Number of major vessels (ca)
5. ST depression (oldpeak)

### 4. Model Interpretability

- **Feature Importance**: LightGBM gain-based importance ranking
- **SHAP Values**: Identified the contribution of each feature to individual predictions

## Technologies Used

- **Python 3.10**
- **Data Manipulation**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Feature Selection**: PySwarms, pymRMR
- **Model Interpretation**: SHAP
- **Statistical Analysis**: SciPy

## Results

The project successfully demonstrates that:
1. LightGBM achieves the best overall performance with 98.1% F1-Score
2. Random Forest and MLP also show excellent results (>97% F1-Score)
3. Feature selection using PSO slightly outperforms mRMR
4. The top 5 features (cp, thal, age, ca, oldpeak) capture most predictive power
5. All models benefit significantly from hyperparameter tuning via Grid Search

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Heart_Disease_Predcition

# Install required packages
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm xgboost pyswarms pymrmr shap scipy
```

## Usage

Run the notebooks in the following order:

1. **EDA.ipynb**: Explore the dataset and visualize relationships
2. **models.ipynb**: Train and evaluate machine learning models
3. **feature_selection.ipynb**: Compare feature selection methods and analyze feature importance

## Acknowledgments

This project was completed as part of the Data Mining and Knowledge Discovery course. The dataset is publicly available on Kaggle and aggregates heart disease data from multiple medical institutions.

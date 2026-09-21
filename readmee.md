# Titanic Survival Prediction

## Overview

This project applies Machine Learning classification algorithms to predict whether a passenger survived the Titanic disaster.

The target variable is:

- `Survived = 0`: Did not survive
- `Survived = 1`: Survived

The project focuses on understanding the complete Machine Learning workflow, including data preprocessing, model training, evaluation, and cross-validation.

---

## Dataset

- 891 passengers
- 12 original columns
- Target: `Survived`

### Features used

For the baseline models, the following features were used:

- `Pclass`
- `Sex`
- `Age`
- `SibSp`
- `Parch`
- `Fare`
- `Embarked`

The following columns were excluded from the baseline:

- `PassengerId`
- `Name`
- `Ticket`
- `Cabin`

`Name`, `Ticket`, and `Cabin` may contain useful information, but they were left out of the baseline to keep the preprocessing simple.

---

## Data Preprocessing

The dataset contains missing values, especially in `Age` and `Embarked`.

### Numerical features

Missing numerical values were filled using the median:

- `Age`
- `SibSp`
- `Parch`
- `Fare`

### Categorical features

The categorical features were processed using:

- `SimpleImputer(strategy="most_frequent")`
- `OneHotEncoder(handle_unknown="ignore")`

### Pipeline

`ColumnTransformer` and `Pipeline` were used to combine preprocessing and model training into a single workflow.

After preprocessing:

- Training set: `712 × 12`
- Test set: `179 × 12`

---

## Models

The following classification models were tested:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier
5. XGBoost Classifier

---

## Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- 5-Fold Cross-Validation

---

## Results

Mean 5-Fold Cross-Validation Accuracy:

| Model | Mean CV Accuracy |
|---|---:|
| Logistic Regression | 0.7978 |
| Decision Tree (`max_depth=5`) | 0.8161 |
| Random Forest | 0.7909 |
| Gradient Boosting | 0.8133 |
| XGBoost | 0.8245 |

### XGBoost

- Test Accuracy: ~0.7989
- Mean CV Accuracy: ~0.8245
- CV Standard Deviation: ~0.0341

---

## What I Learned

Through this project, I practiced:

- Exploratory Data Analysis
- Train/Test Split
- Handling Missing Values
- Numerical and Categorical Features
- One-Hot Encoding
- `ColumnTransformer`
- `Pipeline`
- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting
- XGBoost
- Accuracy, Precision, Recall, and F1-score
- Confusion Matrix
- Cross-Validation
- Overfitting and Underfitting
- Model Comparison

---

## Future Improvements

Possible future improvements include:

- Feature engineering from `Name`
- Extracting information from `Ticket`
- Processing `Cabin`
- Hyperparameter tuning
- Comparing additional evaluation metrics
- Testing different train/test splits
- Trying additional Machine Learning models

---

## Project Structure

```text
titanic-classification/
│
├── Titanic.csv
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
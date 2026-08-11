# Loan Default Prediction

A machine learning project that predicts loan default risk using historical home-equity loan data and a Random Forest classifier.

## Business Problem

Loan defaults create financial risk for lenders. The objective of this project is to identify applicants with a higher probability of default so that potentially high-risk loans can be identified for additional review.

Because defaults represent the minority class, **accuracy alone is not sufficient**. Particular attention is given to **recall for the default class**, since a false negative represents a borrower who defaults but was not identified as high risk by the model.

## Dataset

The analysis uses a Home Equity Loan dataset containing **5,960 observations and 13 variables**, including:

* Loan and mortgage amounts
* Property value
* Delinquent credit lines
* Derogatory credit reports
* Recent credit inquiries
* Debt-to-income ratio
* Credit history
* Employment information

The target variable is `BAD`:

* `0` = Non-default
* `1` = Default

Approximately **20% of the observations are defaults**, creating a class-imbalance problem.

## Methodology

The modeling workflow was designed to reduce data leakage and provide a realistic evaluation of model performance.

### Data Preparation

* Performed exploratory data analysis and evaluated missing values
* Split the raw data into training and test sets **before preprocessing**
* Used median imputation for numerical variables
* Used most-frequent imputation for categorical variables
* Applied one-hot encoding to categorical predictors
* Implemented preprocessing within a Scikit-learn pipeline

### Modeling

A **Random Forest classifier** was trained using class-weight balancing to account for the lower frequency of defaults.

Model performance was evaluated using:

* Precision
* Recall
* F1-score
* Confusion matrix
* ROC curve
* ROC-AUC

## Model Results

| Metric                           |  Result |
| -------------------------------- | ------: |
| Accuracy                         |     92% |
| ROC-AUC                          |    0.96 |
| Default Recall at 0.50 Threshold |     61% |
| Default Recall at 0.35 Threshold | **79%** |

The model demonstrated strong overall discrimination with an ROC-AUC of approximately **0.96**.

However, at the standard 0.50 classification threshold, default recall was only **61%**, meaning a substantial portion of actual defaults were not identified.

## Threshold Optimization

Because failing to identify a potential default can represent significant financial risk, the classification threshold was evaluated rather than relying solely on the standard 0.50 cutoff.

Reducing the decision threshold to **0.35 increased default recall from 61% to 79%** while maintaining strong overall model performance.

This illustrates an important credit-risk modeling principle: the optimal classification threshold should reflect the relative business costs of false negatives and false positives rather than simply maximizing accuracy.

## Key Findings

* Credit behavior variables such as delinquency and derogatory credit history were strongly associated with default risk.
* Random Forest provided strong discrimination between default and non-default borrowers.
* Class imbalance made recall an important metric alongside accuracy.
* Decision-threshold optimization substantially improved identification of borrowers who ultimately defaulted.
* A leakage-safe preprocessing pipeline provides a more reliable estimate of model performance.

## Tools & Technologies

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Random Forest` · `Matplotlib` · `Seaborn` · `Jupyter Notebook`

## Project Notebook

The complete exploratory analysis, preprocessing workflow, model development, evaluation, and threshold analysis are available in:

**`Loan_Default_Prediction.ipynb`**

## Limitations & Next Steps

Potential extensions to this analysis include:

* Comparing Random Forest with logistic regression and gradient-boosting models
* Cross-validation and hyperparameter optimization
* Precision-recall analysis
* Probability calibration
* Model explainability and feature-importance analysis
* Fairness analysis across borrower subgroups
* Cost-based optimization of the classification threshold

---

**Project focus:** Credit Risk · Predictive Modeling · Classification · Machine Learning

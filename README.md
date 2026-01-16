# Loan_Default_Prediction
Predict whether a loan applicant will default using historical loan application data and machine learning models.
Credit Risk Prediction – Random Forest Model
Objective

Predict whether a loan applicant will default using historical loan application data.

Dataset

Home Equity Loan dataset containing applicant demographics, credit attributes, and loan performance.

Approach
Approach

Performed train/test split prior to preprocessing to prevent data leakage

Built Scikit-learn pipeline with:

Median imputation for numerical features

Most-frequent imputation + one-hot encoding for categorical features

Random Forest classifier with class-weight balancing

Evaluated using classification report and ROC-AUC

Tuned decision threshold to improve detection of defaulters

RESULTS: 
Metric	     Value
Accuracy	   92%
ROC-AUC	     0.96
Recall (Default class)	Improved from 61% → 79%

This model has ability to identify high-risk applicants while maintaining strong precision.


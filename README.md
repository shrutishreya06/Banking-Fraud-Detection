# Banking Fraud Detection using Machine Learning

A machine learning project that predicts whether a bank transaction is **fraud (1)** or **not fraud (0)** using an Indian banking transactions dataset. Five classification models are built and compared.

## Problem Statement
Online banking frauds are increasing, but fraud transactions are very few compared to normal ones. The aim of this project is to build models which can predict fraud from the customer, account and transaction details, and to find out which model works best.

## Dataset
- File: `indian_banking_transactions.csv` (not uploaded here because the file is 77 MB)
- 550,000 transactions, 79,916 customers, 20 columns (January 2019 to January 2024)
- Target column: `is_fraud`
- Only **4,873 transactions (0.89%)** are fraud, so the data is highly imbalanced

To run the code, download the dataset and keep it in the same folder as the Python file.

## Project Steps
1. **Data cleaning**: filled missing `loan_type` values with "None", checked duplicates, dropped `transaction_id`, `customer_id`, `transaction_date` and `transaction_time`
2. **Visualization and EDA**: fraud vs not fraud, transaction type vs fraud, amount boxplot, fraud rate by type, correlation heatmap
3. **Preprocessing**: one-hot encoding, Min-Max normalization, stratified sample of 100,000 rows, 80/20 train-test split (`random_state=42`)
4. **Models**: Logistic Regression, Naive Bayes, KNN, Decision Tree, Random Forest
5. **Evaluation**: confusion matrix, accuracy, precision, recall, F1 score and classification report

## Results

| Model | Accuracy | Recall (fraud) |
|---|---|---|
| Logistic Regression | 0.647 | 0.441 |
| Naive Bayes | 0.940 | 0.158 |
| KNN | 0.991 | 0.000 |
| Decision Tree | 0.766 | 0.266 |
| Random Forest | 0.956 | 0.119 |

(Test set: 20,000 transactions, 177 of them are fraud.)

## Key Findings
- Accuracy alone is misleading. KNN has 99.1% accuracy but it did not detect a single fraud.
- **Logistic Regression** found the most frauds among the models with a useful balance (recall 0.441, best ROC AUC 0.573).
- `transaction_amount` is the most important feature. The average fraud amount (INR 119,446) is about 4 times the average normal amount (INR 29,107).
- RTGS transactions have the highest fraud rate (1.85%).
- The precision of all models is low (below 3%), so the dataset has only a weak signal for fraud.

## Plots

**Exploratory plots**

![EDA](images/01_eda_plots.png)

**Confusion matrices of the five models**

![Confusion matrices](images/03_confusion_matrices.png)

**Model specific plots and comparison**

![Model plots](images/04_model_plots_and_comparison.png)

**Feature importance**

![Feature importance](images/06_feature_importance.png)

## How to Run
```bash
pip install -r requirements.txt
python banking_fraud_detection.py
```
The script prints the results and shows all the plots one by one (close each plot window to see the next one). It is easier to run it in Jupyter Notebook.

## Limitations
- Only one train-test split was used, with no cross-validation or hyper-parameter tuning
- Only the default threshold of 0.5 was used
- Methods like SMOTE were not tried
- Most columns have almost the same fraud rate in every category, so the dataset may be simulated

## Future Work
SMOTE or under-sampling, cross-validation, hyper-parameter tuning, features from date/time and customer history, and testing on real bank data.

## Tech Stack
Python, pandas, NumPy, Matplotlib, seaborn, scikit-learn

## Author
**Saruti Shreya**
[LinkedIn profile link] | [Email]

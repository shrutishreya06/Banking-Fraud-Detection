# CA2 : Banking Fraud Detection
# Models : Logistic Regression, Naive Bayes, KNN, Decision Tree, Random Forest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, classification_report

# Problem Statement 
# To predict whether a bank transaction is Fraud (1) or Not Fraud (0)
# Target column : is_fraud

# Load dataset 
df = pd.read_csv("C:\\Users\Saruti Shreya\Downloads\indian_banking_transactions.csv")
print(df.head())
print(df.info())
print(df.describe())

# Data Cleaning 
# identify the NAs in data
print(df.isna().sum())

# loan_type NA means no loan, so fill with "None"
df["loan_type"] = df["loan_type"].fillna("None")

# check duplicates
print("Duplicates:", df.duplicated().sum())

# remove unnecessary columns
df = df.drop(["transaction_id", "customer_id", "transaction_date", "transaction_time"], axis=1)
print(df.isna().sum())

# Visualization 
# Fraud vs Not Fraud
plt.figure()
sns.countplot(x="is_fraud", data=df)
plt.title("Fraud vs Not Fraud")
plt.show()

# Transaction type vs Fraud
plt.figure(figsize=(10, 5))
sns.countplot(x="transaction_type", hue="is_fraud", data=df)
plt.xticks(rotation=45)
plt.title("Transaction Type vs Fraud")
plt.show()

# Amount vs Fraud
plt.figure()
sns.boxplot(x="is_fraud", y="transaction_amount", data=df)
plt.yscale("log")
plt.title("Transaction Amount vs Fraud")
plt.show()

#  EDA and Statistical Analysis
print(df["is_fraud"].value_counts())
print("Fraud %:", round(df["is_fraud"].mean() * 100, 2))

print(df.groupby("transaction_type")["is_fraud"].mean())
print(df.groupby("is_fraud")[["transaction_amount", "account_balance", "credit_score"]].mean())

# correlation analysis
num_cols = ["transaction_amount", "account_balance", "credit_score", "has_loan", "emi_amount", "transaction_hour", 
            "is_fraud"]
print(df[num_cols].corr())
plt.figure(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#Model Development 
# dataset is very big so take 1 lakh rows (fraud ratio remains same)
df, rest = train_test_split(df, train_size=100000, random_state=42, stratify=df["is_fraud"])

# separate features and target
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# convert text columns into numbers
X = pd.get_dummies(X, drop_first=True)

# Min-Max Normalization
scaler = MinMaxScaler()
X[["transaction_amount", "account_balance", "credit_score", "emi_amount"]] = scaler.fit_transform(
    X[["transaction_amount", "account_balance", "credit_score", "emi_amount"]])

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# MODEL 1 : LOGISTIC REGRESSION

lr = LogisticRegression(class_weight="balanced", max_iter=1000)
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

print("\nLogistic Regression")
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_lr))
print("Accuracy:", round(accuracy_score(y_test, pred_lr) * 100, 2), "%")
print(classification_report(y_test, pred_lr))

# Confusion matrix plot
plt.figure()
sns.heatmap(confusion_matrix(y_test, pred_lr), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression - Confusion Matrix")
plt.show()

# Feature coefficients (top 10)
coef = pd.Series(lr.coef_[0], index=X.columns)
top = coef.abs().sort_values(ascending=False).head(10).index
plt.figure()
coef[top].sort_values().plot(kind="barh")
plt.title("Logistic Regression - Feature Coefficients")
plt.show()



# MODEL 2 : NAIVE BAYES

nb = GaussianNB()
nb.fit(X_train, y_train)
pred_nb = nb.predict(X_test)

print("\nNaive Bayes")
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_nb))
print("Accuracy:", round(accuracy_score(y_test, pred_nb) * 100, 2), "%")
print(classification_report(y_test, pred_nb))

# Confusion matrix plot
plt.figure()
sns.heatmap(confusion_matrix(y_test, pred_nb), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Naive Bayes - Confusion Matrix")
plt.show()

# Class-wise mean of numeric features
means = pd.DataFrame(nb.theta_, columns=X.columns, index=["Not Fraud", "Fraud"])
means = means[["transaction_amount", "account_balance", "credit_score", "emi_amount"]]
means.plot(kind="bar")
plt.title("Naive Bayes - Class-wise Mean")
plt.xticks(rotation=0)
plt.show()



# MODEL 3 : KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)

print("\nKNN")
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_knn))
print("Accuracy:", round(accuracy_score(y_test, pred_knn) * 100, 2), "%")
print(classification_report(y_test, pred_knn))

# Confusion matrix plot
plt.figure()
sns.heatmap(confusion_matrix(y_test, pred_knn), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN - Confusion Matrix")
plt.show()

# Effect of K on accuracy and recall
k_values = [1, 3, 5, 7, 9, 11]
acc_list = []
rec_list = []
for k in k_values:
    m = KNeighborsClassifier(n_neighbors=k)
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    acc_list.append(accuracy_score(y_test, p))
    rec_list.append(recall_score(y_test, p))

plt.figure()
plt.plot(k_values, acc_list, marker="o", label="Accuracy")
plt.plot(k_values, rec_list, marker="o", label="Recall")
plt.xlabel("K value")
plt.ylabel("Score")
plt.title("KNN - Effect of K on Accuracy and Recall")
plt.legend()
plt.show()


# MODEL 4 : DECISION TREE

dt = DecisionTreeClassifier(criterion="entropy", max_depth=5, class_weight="balanced", random_state=42)
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)

print("\nDecision Tree")
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_dt))
print("Accuracy:", round(accuracy_score(y_test, pred_dt) * 100, 2), "%")
print(classification_report(y_test, pred_dt))

# Confusion matrix plot
plt.figure()
sns.heatmap(confusion_matrix(y_test, pred_dt), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Decision Tree - Confusion Matrix")
plt.show()

# Tree structure
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=X.columns, class_names=["Not Fraud", "Fraud"], filled=True, max_depth=3)
plt.title("Decision Tree - Tree Structure")
plt.show()

# Feature importance
imp_dt = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
plt.figure()
imp_dt.sort_values().plot(kind="barh")
plt.title("Decision Tree - Feature Importance")
plt.show()


# MODEL 5 : RANDOM FOREST

rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

print("\nRandom Forest")
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_rf))
print("Accuracy:", round(accuracy_score(y_test, pred_rf) * 100, 2), "%")
print(classification_report(y_test, pred_rf))

# Confusion matrix plot
plt.figure()
sns.heatmap(confusion_matrix(y_test, pred_rf), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest - Confusion Matrix")
plt.show()

# Feature importance
imp_rf = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
plt.figure()
imp_rf.sort_values().plot(kind="barh")
plt.title("Random Forest - Feature Importance")
plt.show()


# COMPARISON OF ALL MODELS

comp = pd.DataFrame({
    "Model": ["Logistic Regression", "Naive Bayes", "KNN", "Decision Tree", "Random Forest"],
    "Accuracy": [accuracy_score(y_test, pred_lr), accuracy_score(y_test, pred_nb), accuracy_score(y_test, pred_knn),
                 accuracy_score(y_test, pred_dt), accuracy_score(y_test, pred_rf)],
    "Recall": [recall_score(y_test, pred_lr), recall_score(y_test, pred_nb), recall_score(y_test, pred_knn),
               recall_score(y_test, pred_dt), recall_score(y_test, pred_rf)]
})
print("\nModel Comparison:")
print(comp)

comp.set_index("Model").plot(kind="bar", figsize=(10, 6))
plt.title("Comparison of 5 Models")
plt.xticks(rotation=0)
plt.show()

# Conclusion
# Data is imbalanced (fraud is only about 0.9%), so accuracy alone is not enough.
# Recall of fraud class is also checked. Logistic Regression and Decision Tree
# catch more fraud cases, while KNN has high accuracy but zero fraud recall.
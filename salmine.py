# -- coding: utf-8 --

# Import all the tools we need
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timezone

# Models from Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# Model Evaluations
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("outputs.csv", sep=";")

# Encode suspected_gas
le = LabelEncoder()
df["suspected_gas_encoded"] = le.fit_transform(df["suspected_gas"])

# Features and labels
X = df[["humidity", "mq5", "mq7", "temperature"]]
y = df[["alert", "suspected_gas_encoded"]]  # multi-output y

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MultiOutput Model
multi_model = MultiOutputClassifier(RandomForestClassifier(random_state=42))
multi_model.fit(X_train, y_train)
y_pred = multi_model.predict(X_test)

# Evaluate
print("Accuracy for alert:", accuracy_score(y_test["alert"], y_pred[:, 0]))
print("Accuracy for suspected_gas:", accuracy_score(y_test["suspected_gas_encoded"], y_pred[:, 1]))

# Train/Test for single target (alert) to fit normal classifiers
y_single = y["alert"]
X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(X, y_single, test_size=0.2, random_state=42)

# Put models in a dictionary
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    np.random.seed(42)
    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_scores[name] = model.score(X_test, y_test)
    return model_scores

# Train and score models
model_scores = fit_and_score(models=models,
                             X_train=X_train_single,
                             X_test=X_test_single,
                             y_train=y_train_single,
                             y_test=y_test_single)

print(model_scores)

# Plot model comparison
model_compare = pd.DataFrame(model_scores, index=["Accuracy"])
model_compare.T.plot.bar()
plt.xticks(rotation=0)
plt.title("Model Comparison on 'alert'")
plt.show()

# KNN tuning
train_scores = []
test_scores = []
neighbors = range(1, 21)
knn = KNeighborsClassifier()

for i in neighbors:
    knn.set_params(n_neighbors=i)
    knn.fit(X_train_single, y_train_single)
    train_scores.append(knn.score(X_train_single, y_train_single))
    test_scores.append(knn.score(X_test_single, y_test_single))

plt.plot(neighbors, train_scores, label="Train score")
plt.plot(neighbors, test_scores, label="Test score")
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend()
plt.title("KNN tuning")
plt.show()

print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")

# Hyperparameter tuning example for Logistic Regression
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

np.random.seed(42)

rs_log_reg = RandomizedSearchCV(LogisticRegression(max_iter=1000),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)

rs_log_reg.fit(X_train_single, y_train_single)

print(f"Best parameters for Logistic Regression: {rs_log_reg.best_params_}")
print(f"Best score: {rs_log_reg.best_score_}")

# Save the multi-output model
joblib.dump(multi_model, "alert_model.pkl")
print("Model saved as alert_model.pkl")

print("\nAnalysis completed successfully!")
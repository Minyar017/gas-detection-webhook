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
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -------------------- Load and Prepare Data --------------------

df = pd.read_csv("gas_data1.csv", sep=";")

# Keep only alert values 0 and 1
df = df[df["alert"].isin([0, 1])]

# Encode suspected_gas
le = LabelEncoder()
df["suspected_gas_encoded"] = le.fit_transform(df["suspected_gas"])

# -------------------- EDA (Exploration des données) --------------------

# Classe balance
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="alert")
plt.title("Répartition des classes (alert)")
plt.xlabel("Alert (0 = Normal, 1 = Danger)")
plt.ylabel("Nombre d'exemples")
plt.tight_layout()
plt.show()

# Heatmap des corrélations
plt.figure(figsize=(8, 6))
sns.heatmap(df[["humidity", "mq5", "mq7", "temperature", "alert"]].corr(), annot=True, cmap="coolwarm")
plt.title("Corrélation entre les variables")
plt.show()

# -------------------- Features and Labels --------------------

X = df[["humidity", "mq5", "mq7", "temperature"]]
y = df[["alert", "suspected_gas_encoded"]]

# Normalisation
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- Multi-Output Model (RandomForest) --------------------

multi_model = MultiOutputClassifier(RandomForestClassifier(random_state=42))
multi_model.fit(X_train, y_train)
y_pred = multi_model.predict(X_test)

# Accuracy
print("Accuracy - Alert:", accuracy_score(y_test["alert"], y_pred[:, 0]))
print("Accuracy - Suspected Gas:", accuracy_score(y_test["suspected_gas_encoded"], y_pred[:, 1]))

# Confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test["alert"], y_pred[:, 0]), annot=True, fmt="d", ax=ax[0])
ax[0].set_title("Confusion Matrix - Alert")
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("True")

sns.heatmap(confusion_matrix(y_test["suspected_gas_encoded"], y_pred[:, 1]), annot=True, fmt="d", ax=ax[1])
ax[1].set_title("Confusion Matrix - Suspected Gas")
ax[1].set_xlabel("Predicted")
ax[1].set_ylabel("True")

plt.tight_layout()
plt.show()

# -------------------- Feature Importance --------------------

importances = multi_model.estimators_[0].feature_importances_
features = ["humidity", "mq5", "mq7", "temperature"]

plt.figure(figsize=(6, 4))
sns.barplot(x=importances, y=features)
plt.title("Importance des caractéristiques (Alert)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# -------------------- ROC Curve for 'alert' --------------------

fpr, tpr, _ = roc_curve(y_test["alert"], y_pred[:, 0])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Alert")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------- Model Benchmarking --------------------

# Training only on 'alert'
y_single = y["alert"]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_single, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier()
}

def fit_and_score(models, X_train, X_test, y_train, y_test):
    np.random.seed(42)
    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        scores[name] = model.score(X_test, y_test)
    return scores

model_scores = fit_and_score(models, X_train_s, X_test_s, y_train_s, y_test_s)

# Affichage des scores
print("Model Scores:")
for name, score in model_scores.items():
    print(f"{name}: {score:.2f}")

# Bar plot
pd.DataFrame(model_scores, index=["Accuracy"]).T.plot.bar(legend=False)
plt.ylabel("Accuracy")
plt.title("Comparaison des modèles sur 'alert'")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# -------------------- Cross-validation --------------------

rf = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(rf, X, y_single, cv=5)
print(f"Cross-Validation Accuracy (Random Forest): {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")

# -------------------- KNN Tuning --------------------

train_scores, test_scores = [], []
neighbors = range(1, 21)

for i in neighbors:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_s, y_train_s)
    train_scores.append(knn.score(X_train_s, y_train_s))
    test_scores.append(knn.score(X_test_s, y_test_s))

plt.plot(neighbors, train_scores, label="Train")
plt.plot(neighbors, test_scores, label="Test")
plt.xlabel("Nombre de voisins")
plt.ylabel("Score")
plt.title("KNN Tuning")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------- Hyperparameter Tuning (Logistic Regression) --------------------

param_grid = {"C": np.logspace(-4, 4, 20), "solver": ["liblinear"]}
search = RandomizedSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, n_iter=20, verbose=0, random_state=42)
search.fit(X_train_s, y_train_s)

print(f"Best Logistic Regression Params: {search.best_params_}")
print(f"Best CV Score: {search.best_score_:.2f}")

# -------------------- Save Model --------------------

joblib.dump(multi_model, "alert_model.pkl")
print("✅ Modèle sauvegardé : alert_model.pkl")

print("\n✅ Analyse complète terminée avec succès.")

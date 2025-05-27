# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath, sep=';')
    df = df[df['alert'].isin([0, 1])]
    le = LabelEncoder()
    df['suspected_gas_encoded'] = le.fit_transform(df['suspected_gas'])
    return df, le

def exploratory_data_analysis(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='alert', data=df)
    plt.title('Répartition des classes (Alert)')
    plt.xlabel('Alert (0 = Normal, 1 = Danger)')
    plt.ylabel("Nombre d'exemples")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    corr = df[['humidity', 'mq5', 'mq7', 'temperature', 'alert']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Matrice de corrélation')
    plt.tight_layout()
    plt.show()

def preprocess_features_labels(df):
    X = df[['humidity', 'mq5', 'mq7', 'temperature']].values
    y = df[['alert', 'suspected_gas_encoded']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def train_multioutput_model(X_train, y_train):
    # Modèle plus régularisé pour éviter l'overfitting
    multi_model = MultiOutputClassifier(RandomForestClassifier(
        n_estimators=50,  
        max_depth=4,      
        min_samples_leaf=10, 
        min_samples_split=20, 
        max_features='sqrt',
        random_state=42))
    multi_model.fit(X_train, y_train)
    return multi_model

def evaluate_multioutput_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc_alert = accuracy_score(y_test['alert'], y_pred[:, 0])
    acc_gas = accuracy_score(y_test['suspected_gas_encoded'], y_pred[:, 1])
    print(f"Accuracy (Alert): {acc_alert:.4f}")
    print(f"Accuracy (Suspected Gas): {acc_gas:.4f}")

def detect_anomalies_with_isolationforest(X_train, X_test):
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    iso_forest.fit(X_train)
    preds = iso_forest.predict(X_test)

    plt.figure(figsize=(6, 4))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=preds, cmap='coolwarm', edgecolor='k')
    plt.title("Détection d'anomalies avec IsolationForest")
    plt.xlabel('Humidité')
    plt.ylabel('MQ5')
    plt.tight_layout()
    plt.show()

def optimize_hyperparameters(X_train, y_train, target_name):
    """
    Optimise les hyperparamètres pour réduire l'overfitting
    """
    optimized_classifiers = {}
    
    # 1. LOGISTIC REGRESSION avec régularisation
    print(f"\n🔍 Optimisation Logistic Regression pour {target_name}...")
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=2000))
    ])
    
    lr_param_grid = {
        'classifier__C': [0.01, 0.1, 0.5, 1.0, 2.0],  
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear', 'saga']
    }
    
    lr_grid = GridSearchCV(lr_pipeline, lr_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    optimized_classifiers['Logistic Regression'] = lr_grid.best_estimator_
    print(f"✅ Meilleurs paramètres LR: {lr_grid.best_params_}")
    
    # 2. KNN avec optimisation
    print(f"\n🔍 Optimisation KNN pour {target_name}...")
    knn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier())
    ])
    
    knn_param_grid = {
        'classifier__n_neighbors': [5, 7, 9, 11, 15, 19],  
        'classifier__weights': ['uniform', 'distance'],
        'classifier__metric': ['euclidean', 'manhattan']
    }
    
    knn_grid = GridSearchCV(knn_pipeline, knn_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    knn_grid.fit(X_train, y_train)
    optimized_classifiers['K-Nearest Neighbors'] = knn_grid.best_estimator_
    print(f"✅ Meilleurs paramètres KNN: {knn_grid.best_params_}")
    
    # 3. NAIVE BAYES avec lissage
    print(f"\n🔍 Optimisation Naive Bayes pour {target_name}...")
    nb_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GaussianNB())
    ])
    
    nb_param_grid = {
        'classifier__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    }
    
    nb_grid = GridSearchCV(nb_pipeline, nb_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    nb_grid.fit(X_train, y_train)
    optimized_classifiers['Naive Bayes'] = nb_grid.best_estimator_
    print(f"✅ Meilleurs paramètres NB: {nb_grid.best_params_}")
    
    return optimized_classifiers

def plot_combined_learning_curves(classifiers, X, y, title):
    """
    Trace les courbes d'apprentissage avec plus de points pour mieux voir l'overfitting
    """
    plt.figure(figsize=(12, 8))
    
    for name, estimator in classifiers.items():
        # Plus de points pour mieux visualiser
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10), 
            scoring='accuracy')

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Tracer avec zones d'incertitude
        plt.plot(train_sizes, train_mean, '--', label=f'{name} - Train', linewidth=2)
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        
        plt.plot(train_sizes, test_mean, '-', label=f'{name} - Test', linewidth=2)
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)

   
    plt.xlabel("Taille du jeu d'entraînement", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def train_and_evaluate_classifiers(X_train, X_test, y_train, y_test, target_name, label_encoder=None):
    print(f"\n{'='*50}")
    print(f"🚀 OPTIMISATION POUR {target_name.upper()}")
    print(f"{'='*50}")
    
    # Utiliser les modèles optimisés
    classifiers = optimize_hyperparameters(X_train, y_train, target_name)

    scores = {}
    confusion_matrices = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 

    for name, clf in classifiers.items():
        print(f"\n📊 Évaluation {name}...")
        
        # Validation croisée
        cv_scores = cross_val_score(clf, X_train, y_train, cv=skf)
        print(f"{name} ({target_name}) - Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Test final
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        scores[name] = acc
        
        print(f"{name} ({target_name}) - Test Accuracy: {acc:.4f}")
        
        # Vérification de l'overfitting
        train_acc = clf.score(X_train, y_train)
        gap = train_acc - acc
        print(f"🔍 Train Accuracy: {train_acc:.4f} | Gap (Train-Test): {gap:.4f}")
        
        if gap > 0.02:  # Si l'écart > 2%
            print("⚠️  Possible overfitting détecté!")
        else:
            print("✅ Pas d'overfitting significatif")
        
        print(classification_report(y_test, y_pred, zero_division=0))
        
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = cm

    # Matrices de confusion
    if label_encoder and target_name == "suspected_gas":
        labels = label_encoder.inverse_transform(np.unique(y_test))
    else:
        labels = np.unique(y_test)

    fig, axes = plt.subplots(1, len(classifiers), figsize=(20, 5))
    for ax, (name, cm) in zip(axes, confusion_matrices.items()):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f"{name} (Optimisé)")
    plt.suptitle(f"Matrices de confusion - {target_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Comparaison des scores
    plt.figure(figsize=(10, 6))
    bars = plt.bar(list(scores.keys()), list(scores.values()), 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title(f"Modèles  - Accuracy sur '{target_name}'", fontsize=14)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0.9, 1.0)  # Zoom sur la zone d'intérêt
    plt.xticks(rotation=45)
    
    # Ajouter les valeurs sur les barres
    for bar, score in zip(bars, scores.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{score:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

    # Courbes d'apprentissage optimisées
    plot_combined_learning_curves(classifiers, X_train, y_train, f"Courbes d'apprentissage - {target_name}")

    return classifiers, scores

def save_all_models(classifiers_dict):
    joblib.dump(classifiers_dict, 'alert_model.pkl')
    print("✅ Tous les modèles optimisés ont été sauvegardés dans 'alert_model.pkl'.")

def main():
    print("🔥 DÉTECTION DE GAZ - VERSION ANTI-OVERFITTING")
    print("="*60)
    
    df, label_encoder = load_and_prepare_data("alerted_gas.csv")
    exploratory_data_analysis(df)

    X, y, scaler = preprocess_features_labels(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y['alert'], random_state=42)

    # Modèle multi-sortie optimisé
    print("\n🤖 Entraînement du modèle multi-sortie optimisé...")
    multi_model = train_multioutput_model(X_train, y_train)
    evaluate_multioutput_model(multi_model, X_test, y_test)

    # Anomalies
    detect_anomalies_with_isolationforest(X_train, X_test)

    # Modèles optimisés pour 'alert'
    y_alert_train = y_train['alert']
    y_alert_test = y_test['alert']
    classifiers_alert, scores_alert = train_and_evaluate_classifiers(
        X_train, X_test, y_alert_train, y_alert_test, 'alert'
    )

    # Modèles optimisés pour 'suspected_gas'
    y_gas_train = y_train['suspected_gas_encoded']
    y_gas_test = y_test['suspected_gas_encoded']
    classifiers_gas, scores_gas = train_and_evaluate_classifiers(
        X_train, X_test, y_gas_train, y_gas_test, 'suspected_gas', label_encoder
    )

    all_models = {
        'multi_output_model': multi_model,
        'classifiers_alert': classifiers_alert,
        'classifiers_suspected_gas': classifiers_gas,
        'scaler': scaler,
        'label_encoder': label_encoder
    }

    save_all_models(all_models)
    print("\n🎉 Traitement terminé avec succès - Modèles optimisés contre l'overfitting!")

if __name__ == "__main__":
    main()
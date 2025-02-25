import numpy as np
import pandas as pd
import warnings
import time
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (recall_score, roc_auc_score, precision_score,
                             f1_score, accuracy_score, make_scorer)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


warnings.filterwarnings("ignore")  # Suppress warnings


def find_best_model_for_recall(X_train, y_train, X_test, y_test, param_grids, classifiers, cv=5, n_iter=20, random_state=42):
    """
    Optimizes hyperparameters to maximize recall for class 1 and stores results in a DataFrame.

    Parameters:
    - X_train, y_train: Training data
    - X_test, y_test: Testing data
    - param_grids: Dictionary of hyperparameter grids
    - classifiers: Dictionary of models
    - cv: Number of cross-validation folds
    - n_iter: Number of random search iterations per model
    - random_state: Random state for reproducibility

    Returns:
    - Best model, best parameters, best recall score
    - DataFrame containing recall, ROC AUC, precision, F1-score, accuracy, and best hyperparameters for each model
    """

    results_list = []
    best_model = None
    best_params = None
    best_recall = 0
    best_model_name = None

    # Optimize for recall of class 1
    scoring = make_scorer(recall_score, average="binary", pos_label=1)
    cv_strategy = StratifiedKFold(
        n_splits=cv, shuffle=True, random_state=random_state)

    for model_name, model in classifiers.items():
        if model_name not in param_grids:
            continue

        print(f"\n🔍 Tuning {model_name}...")
        param_grid = param_grids[model_name]

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv_strategy,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )

        start_time = time.time()
        random_search.fit(X_train, y_train)
        elapsed_time = time.time() - start_time

        best_estimator = random_search.best_estimator_
        best_hyperparams = random_search.best_params_

        # Predictions on the test set
        y_pred = best_estimator.predict(X_test)
        y_prob = best_estimator.predict_proba(X_test)[:, 1] if hasattr(
            best_estimator, "predict_proba") else best_estimator.decision_function(X_test)

        # Compute evaluation metrics
        recall = recall_score(y_test, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(
            y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"✅ {model_name} - Recall: {recall:.4f}, ROC AUC: {roc_auc:.4f}, Precision: {precision:.4f}, F1-score: {f1:.4f}, Accuracy: {accuracy:.4f} | Time: {elapsed_time:.2f}s")

        # Store results
        results_list.append({
            "Model": model_name,
            "Best Recall": recall,
            "ROC AUC": roc_auc,
            "Precision": precision,
            "F1-score": f1,
            "Accuracy": accuracy,
            "Best Parameters": best_hyperparams
        })

        # Update best model if current one is better
        if recall > best_recall:
            best_model = best_estimator
            best_params = best_hyperparams
            best_recall = recall
            best_model_name = model_name

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list).sort_values(
        by="Best Recall", ascending=False)

    print("\n🏆 Best Model Found:")
    print(f"📌 Model: {best_model_name}")
    print(f"📊 Best Recall Score: {best_recall:.4f}")
    print(f"⚙️ Best Hyperparameters: {best_params}")

    return best_model, best_params, best_recall, results_df

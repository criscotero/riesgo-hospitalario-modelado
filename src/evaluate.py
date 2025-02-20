import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Example function to evaluate models


def evaluate_models(models, param_grids, X_train, X_test, y_train, y_test):
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        # Perform Grid Search with Cross-Validation
        grid_search = GridSearchCV(
            model, param_grids[name], cv=cv, scoring='recall', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(
            best_model, "predict_proba") else None

        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        auc_roc = roc_auc_score(
            y_test, y_pred_proba) if y_pred_proba is not None else np.nan

        results.append({
            'Model': name,
            'Best Params': grid_search.best_params_,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC-ROC': auc_roc
        })

    results_df = pd.DataFrame(results)
    return results_df.sort_values(by='Recall', ascending=False)

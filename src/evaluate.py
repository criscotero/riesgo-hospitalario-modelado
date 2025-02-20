import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Example function to evaluate models


import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_models(models, param_grids, X_train, X_test, y_train, y_test, save_path="models"):
    """Evaluate models using Grid Search, save results, and store trained models for production."""

    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"Training and tuning {name}...")

        # Perform Grid Search with Cross-Validation
        grid_search = GridSearchCV(
            model, param_grids.get(name, {}), cv=cv, scoring='recall', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(
            best_model, "predict_proba") else None

        # Compute evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(
            y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        auc_roc = roc_auc_score(
            y_test, y_pred_proba) if y_pred_proba is not None else np.nan

        # Append results
        results.append({
            'Model': name,
            'Best Params': grid_search.best_params_,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC-ROC': auc_roc
        })

        # Save the trained model
        model_filename = os.path.join(
            save_path, f"{name.replace(' ', '_')}.pkl")
        joblib.dump(best_model, model_filename)
        print(f"Saved {name} model to {model_filename}")

    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(save_path, "results.csv")
    results_df.to_csv(results_csv, index=False)

    print(f"\nAll results saved to {results_csv}")

    return results_df.sort_values(by='Recall', ascending=False)

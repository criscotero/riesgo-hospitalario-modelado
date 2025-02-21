import os
import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def identify_feature_types(df, target_column):
    """Identify categorical and numerical columns in the dataframe."""
    categorical_cols = df.select_dtypes(
        include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(
        include=['int64', 'float64']).columns.tolist()

    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)

    return categorical_cols, numerical_cols


def create_preprocessing_pipeline(categorical_cols, numerical_cols):
    """Create a preprocessing pipeline for categorical and numerical features."""
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="constant", fill_value="Missing")),
        ('encoder', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('cat', categorical_pipeline, categorical_cols),
        ('num', numerical_pipeline, numerical_cols)
    ])

    return preprocessor


def preprocess_and_split(df, target_column, test_size=0.2, random_state=42):
    """Preprocess data and split into training and testing sets."""
    categorical_cols, numerical_cols = identify_feature_types(
        df, target_column)
    preprocessor = create_preprocessing_pipeline(
        categorical_cols, numerical_cols)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Transform data
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    # Preserve original column names instead of `cat__` and `num__` prefixes
    feature_names = categorical_cols + numerical_cols

    # Convert to DataFrame
    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=feature_names)
    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=feature_names)

    return X_train_encoded_df, X_test_encoded_df, y_train, y_test, preprocessor


def apply_smote(X_train, y_train, random_state=42, minority_ratio=1.0):
    """
    Apply SMOTE to oversample only class 1.

    Parameters:
    - X_train (pd.DataFrame or np.array): Training feature set.
    - y_train (pd.Series or np.array): Training labels.
    - random_state (int): Random state for reproducibility.
    - minority_ratio (float): The desired ratio of class 1 samples to class 0 samples.

    Returns:
    - X_train_resampled: The resampled feature set.
    - y_train_resampled: The resampled labels.
    """
    from collections import Counter

    # Get class distribution
    class_counts = Counter(y_train)
    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)

    majority_count = class_counts[majority_class]
    new_minority_count = int(majority_count * minority_ratio)

    sampling_strategy = {minority_class: new_minority_count}

    print(f"Original class distribution: {class_counts}")
    print(
        f"Applying SMOTE with target class distribution: {sampling_strategy}")

    smote = SMOTE(sampling_strategy=sampling_strategy,
                  random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    new_counts = Counter(y_train_resampled)
    print(f"New class distribution: {new_counts}")

    return X_train_resampled, y_train_resampled


def select_features_mutual_info(X_train, y_train, k=50):
    """Select the top K features based on Mutual Information."""
    k = min(k, X_train.shape[1])
    mi_selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_selected = mi_selector.fit_transform(X_train, y_train)

    selected_features = np.array(X_train.columns)[
        mi_selector.get_support()].tolist()

    print("\nSelected features after Mutual Information:", selected_features)
    return X_train_selected, selected_features, mi_selector


def select_features_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """Select features using a Random Forest model-based selection."""
    rf_selector = SelectFromModel(RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state), threshold="median")

    X_train_final = rf_selector.fit_transform(X_train, y_train)

    selected_indices = rf_selector.get_support(indices=True)
    final_selected_features = np.array(X_train.columns)[
        selected_indices].tolist()

    print("\nSelected features after Random Forest:", final_selected_features)
    return X_train_final, final_selected_features, rf_selector


def save_pipeline(pipeline, preprocessor, mi_selector, rf_selector, save_path):
    """Save pipeline and preprocessing steps for future use."""
    os.makedirs(save_path, exist_ok=True)

    joblib.dump(preprocessor, os.path.join(save_path, "preprocessor.pkl"))
    joblib.dump(mi_selector, os.path.join(save_path, "mi_selector.pkl"))
    joblib.dump(rf_selector, os.path.join(save_path, "rf_selector.pkl"))
    joblib.dump(pipeline, os.path.join(save_path, "pipeline.pkl"))


def create_feature_selection_pipeline(df, target_column, mi_k=50, rf_n_estimators=100, test_size=0.2, random_state=42, save_path="model_assets"):
    """
    Main function to create a feature selection pipeline.
    """
    print("Preprocessing and splitting dataset...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(
        df, target_column, test_size, random_state)

    print("\nColumns after preprocessing:", X_train.columns.tolist())

    print("Applying SMOTE...")
    X_train_resampled, y_train_resampled = apply_smote(
        X_train, y_train, random_state)

    print("Selecting features using Mutual Information...")
    X_train_selected, selected_features, mi_selector = select_features_mutual_info(
        X_train_resampled, y_train_resampled, mi_k)

    X_train_selected_df = pd.DataFrame(
        X_train_selected, columns=selected_features)

    print("Selecting features using Random Forest model...")
    X_train_final, final_selected_features, rf_selector = select_features_random_forest(
        X_train_selected_df, y_train_resampled, rf_n_estimators, random_state)

    valid_final_features = [
        col for col in final_selected_features if col in X_train_selected_df.columns]

    if not valid_final_features:
        raise ValueError(
            "No valid features found for training. Check feature selection steps.")

    X_train_final_df = X_train_selected_df[valid_final_features]

    print("\nFinal selected features for training:", valid_final_features)

    print("Training final pipeline...")
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('mi_selection', mi_selector),
        ('rf_selection', rf_selector),
        ('classifier', RandomForestClassifier(
            n_estimators=rf_n_estimators, random_state=random_state))
    ])

    pipeline.fit(X_train_final_df, y_train_resampled)

    print("Saving pipeline and feature selection models...")
    save_pipeline(pipeline, preprocessor, mi_selector, rf_selector, save_path)

    feature_importance_values = rf_selector.estimator_.feature_importances_[
        :len(valid_final_features)]
    feature_description = pd.DataFrame({
        'Feature': valid_final_features,
        'Importance': feature_importance_values
    }).sort_values(by='Importance', ascending=False)

    print("Pipeline training completed.")
    return pipeline, X_train, X_test, y_train, y_test, feature_description

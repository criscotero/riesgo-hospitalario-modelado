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


def create_feature_selection_pipelinex(df, target_column, mi_k=50, rf_n_estimators=100, test_size=0.2, random_state=42, save_path="model_assets"):
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


def create_feature_selection_pipeline(df: pd.DataFrame, target_column: str,
                                      mi_k: int = 100, rf_n_estimators: int = 100,
                                      test_size: float = 0.2, random_state: int = 42,
                                      save_path: str = "model_assets"):
    """
    Creates a feature selection pipeline with SMOTE for class 1 oversampling.
    """

    print("\n🚀 Starting Feature Selection Pipeline...\n")
    os.makedirs(save_path, exist_ok=True)

    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(
        include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(
        include=['int64', 'float64']).columns.tolist()

    categorical_cols = [
        col for col in categorical_cols if col != target_column]
    numerical_cols = [col for col in numerical_cols if col != target_column]

    print(
        f"🔹 Found {len(categorical_cols)} categorical features and {len(numerical_cols)} numerical features.")

    # Define preprocessing steps
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

    # Feature selection steps
    mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(
        mi_k, len(categorical_cols + numerical_cols)))
    rf_selector = SelectFromModel(RandomForestClassifier(
        n_estimators=rf_n_estimators, random_state=random_state), threshold="median")

    # Full pipeline (without SMOTE yet)
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('mi_selection', mi_selector),
        ('rf_selection', rf_selector),
        ('classifier', RandomForestClassifier(
            n_estimators=rf_n_estimators, random_state=random_state))
    ])

    # Split dataset
    print("📊 Splitting dataset into training and testing sets...")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    print(
        f"✅ Data split complete: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.")

    # Apply preprocessing before SMOTE
    print("⚙️ Applying preprocessing transformation...")
    preprocessor = pipeline.named_steps['preprocessing']
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Get transformed feature names from ColumnTransformer
    transformed_feature_names = preprocessor.get_feature_names_out()
    print(
        f"✅ Preprocessing complete. Transformed feature names:\n{transformed_feature_names}")

    # Apply SMOTE to training data only
    print("⚖️ Applying SMOTE to balance class distribution in training set...")
    smote = SMOTE(sampling_strategy={1: int(
        sum(y_train == 0))}, random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_preprocessed, y_train)

    print(
        f"✅ SMOTE applied: {sum(y_train_resampled == 0)} class 0 samples, {sum(y_train_resampled == 1)} class 1 samples.")

    # Convert to DataFrame for feature selection
    X_train_resampled_df = pd.DataFrame(
        X_train_resampled, columns=transformed_feature_names)

    # Fit the feature selection pipeline (excluding preprocessing)
    print("🚀 Fitting the feature selection pipeline...")
    pipeline.steps = pipeline.steps[1:]  # Remove preprocessing step
    pipeline.fit(X_train_resampled_df, y_train_resampled)
    print("✅ Pipeline training complete.")

    # Extract selected feature names after Mutual Information selection
    print("📊 Extracting selected features after Mutual Information selection...")
    mi_support_mask = pipeline.named_steps['mi_selection'].get_support()
    mi_selected_features = transformed_feature_names[mi_support_mask]
    print(
        f"🔹 {len(mi_selected_features)} features selected after Mutual Information:")
    # Print full list of MI-selected features
    print(mi_selected_features.tolist())

    # Extract final selected features after Random Forest selection
    rf_support_mask = pipeline.named_steps['rf_selection'].get_support()
    if len(mi_selected_features) != len(rf_support_mask):
        raise ValueError(
            f"Feature selection step mismatch: {len(mi_selected_features)} MI-selected features, but {len(rf_support_mask)} RF-selected features."
        )

    selected_features = np.array(mi_selected_features)[rf_support_mask]
    print(
        f"🌲 {len(selected_features)} final features selected after Random Forest selection:")
    # Print full list of RF-selected features
    print(selected_features.tolist())

    # Extract feature importances from the trained random forest model
    feature_importance_values = pipeline.named_steps['rf_selection'].estimator_.feature_importances_
    rf_selected_importances = feature_importance_values[rf_support_mask]

    # Keep the preprocessor in the pipeline
    final_pipeline = Pipeline([
        ('preprocessing', preprocessor),  # Add back preprocessing
        ('mi_selection', pipeline.named_steps['mi_selection']),
        ('rf_selection', pipeline.named_steps['rf_selection']),
        ('classifier', pipeline.named_steps['classifier'])  # Keep classifier
    ])

    # Create a dataframe describing selected features
    feature_description = pd.DataFrame({
        'Feature': selected_features,
        'Importance': rf_selected_importances
    }).sort_values(by='Importance', ascending=False)

    print(f"🔍 {len(selected_features)} final selected features and their importance:")
    print(feature_description)

    return final_pipeline, X_train_resampled, X_test_preprocessed, y_train_resampled, y_test, feature_description

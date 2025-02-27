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


def create_feature_selection_pipeline(df: pd.DataFrame, target_column: str,
                                      mi_k: int = 100, rf_n_estimators: int = 100,
                                      test_size: float = 0.2, random_state: int = 42,
                                      save_path: str = "model_assets"):
    """
    Creates a feature selection pipeline with SMOTE for class 1 oversampling.
    Returns Pandas DataFrames instead of NumPy arrays.
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

    # Convert preprocessed data into DataFrame
    X_train_preprocessed_df = pd.DataFrame(
        X_train_preprocessed, columns=transformed_feature_names)
    X_test_preprocessed_df = pd.DataFrame(
        X_test_preprocessed, columns=transformed_feature_names)

    # Apply SMOTE to training data only
    print("⚖️ Applying SMOTE to balance class distribution in training set...")
    smote = SMOTE(sampling_strategy={1: int(
        sum(y_train == 0))}, random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_preprocessed_df, y_train)

    # Convert resampled data into DataFrame
    X_train_resampled_df = pd.DataFrame(
        X_train_resampled, columns=transformed_feature_names)
    y_train_resampled_df = pd.Series(y_train_resampled)

    print(
        f"✅ SMOTE applied: {sum(y_train_resampled == 0)} class 0 samples, {sum(y_train_resampled == 1)} class 1 samples.")

    # Fit the feature selection pipeline (excluding preprocessing)
    print("🚀 Fitting the feature selection pipeline...")
    pipeline.steps = pipeline.steps[1:]  # Remove preprocessing step
    pipeline.fit(X_train_resampled_df, y_train_resampled_df)
    print("✅ Pipeline training complete.")

    # Extract selected feature names after Mutual Information selection
    print("📊 Extracting selected features after Mutual Information selection...")
    mi_support_mask = pipeline.named_steps['mi_selection'].get_support()
    mi_selected_features = transformed_feature_names[mi_support_mask]
    print(
        f"🔹 {len(mi_selected_features)} features selected after Mutual Information:")
    print(mi_selected_features.tolist())

    # Extract final selected features after Random Forest selection
    rf_support_mask = pipeline.named_steps['rf_selection'].get_support()
    if len(mi_selected_features) != len(rf_support_mask):
        raise ValueError(
            f"Feature selection step mismatch: {len(mi_selected_features)} MI-selected features, but {len(rf_support_mask)} RF-selected features.")

    selected_features = np.array(mi_selected_features)[rf_support_mask]
    print(
        f"🌲 {len(selected_features)} final features selected after Random Forest selection:")
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

    return final_pipeline, X_train_resampled_df, X_test_preprocessed_df, y_train_resampled_df, y_test, feature_description

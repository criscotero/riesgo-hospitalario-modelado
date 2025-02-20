from imblearn.under_sampling import RandomUnderSampler
import joblib
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def select_top_features_tree(df: pd.DataFrame, target: str, n_features: int = 20):
    """
    Selects the top `n_features` most important features using a Random Forest.

    Args:
        df (pd.DataFrame): The dataset containing features and target.
        target (str): The target variable name.
        n_features (int): Number of features to select.

    Returns:
        List of selected feature names.
    """
    X = df.drop(columns=[target])  # Features
    y = df[target]  # Target variable

    # Train a Random Forest to compute feature importance
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Get feature importance and sort
    feature_importances = pd.Series(
        model.feature_importances_, index=X.columns)
    top_features = feature_importances.nlargest(n_features).index.tolist()

    return top_features


def select_top_features_mi(df: pd.DataFrame, target: str, n_features: int = 20):
    """
    Selects the top `n_features` based on Mutual Information.

    Args:
        df (pd.DataFrame): The dataset containing features and target.
        target (str): The target variable name.
        n_features (int): Number of features to select.

    Returns:
        List of selected feature names.
    """
    X = df.drop(columns=[target])  # Features
    y = df[target]  # Target variable

    mi_scores = mutual_info_classif(X, y)
    mi_series = pd.Series(mi_scores, index=X.columns)

    top_features = mi_series.nlargest(n_features).index.tolist()
    return top_features


def create_feature_selection_pipeline(df, target_column, mi_k=50, rf_n_estimators=100, test_size=0.2, random_state=42, save_path="model_assets"):
    """
    Creates a feature selection pipeline that:
    1. Handles missing values by imputing categorical and numerical features.
    2. Encodes categorical features using one-hot encoding.
    3. Scales numerical features using standardization.
    4. Performs feature selection using Mutual Information and Random Forest-based selection.
    5. Balances the dataset using Random Undersampling.
    6. Trains a Random Forest classifier as a placeholder model.
    7. Returns selected feature descriptions and saves encoders/scalers for production deserialization.

    Parameters:
    - df (pd.DataFrame): Input dataset.
    - target_column (str): Name of the target column.
    - mi_k (int): Number of top features to keep after Mutual Information selection. Default is 50.
    - rf_n_estimators (int): Number of estimators for the Random Forest model. Default is 100.
    - test_size (float): Proportion of data to use for testing. Default is 0.2.
    - random_state (int): Random state for reproducibility. Default is 42.
    - save_path (str): Path to save encoders and scalers for model deserialization. Default is "model_assets".

    Returns:
    - pipeline (Pipeline): The trained feature selection pipeline.
    - X_train (pd.DataFrame): Training feature set.
    - X_test (pd.DataFrame): Testing feature set.
    - y_train (pd.Series): Training labels.
    - y_test (pd.Series): Testing labels.
    - feature_description (pd.DataFrame): Dataframe with selected feature statistics (min, max, unique values).
    """
    import os
    os.makedirs(save_path, exist_ok=True)

    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(
        include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(
        include=['int64', 'float64']).columns.tolist()

    # Ensure target column is not in feature list
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)

    # Handle missing values by adding a new category for categorical features
    categorical_imputer = SimpleImputer(
        strategy="constant", fill_value="Missing")
    numerical_imputer = SimpleImputer(strategy="mean")

    # One-hot encode categorical features
    categorical_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1)

    # Standardize numerical features
    numerical_scaler = StandardScaler()

    # Create preprocessing pipelines
    categorical_pipeline = Pipeline([
        ('imputer', categorical_imputer),
        ('encoder', categorical_encoder)
    ])

    numerical_pipeline = Pipeline([
        ('imputer', numerical_imputer),
        ('scaler', numerical_scaler)
    ])

    # Combine preprocessors
    preprocessor = ColumnTransformer([
        ('cat', categorical_pipeline, categorical_cols),
        ('num', numerical_pipeline, numerical_cols)
    ])

    # Split dataset
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # Apply Random Undersampling to balance the dataset
    undersampler = RandomUnderSampler(
        sampling_strategy='auto', random_state=random_state)
    X_train_resampled, y_train_resampled = undersampler.fit_resample(
        X_train, y_train)

    # Transform data with preprocessing pipeline
    X_train_encoded = preprocessor.fit_transform(X_train_resampled)
    feature_names = preprocessor.get_feature_names_out()

    # Convert to DataFrame for compatibility
    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=feature_names)

    # Apply Mutual Information feature selection
    # Ensure we don't select more than available features
    mi_k = min(mi_k, X_train_encoded_df.shape[1])
    mi_selector = SelectKBest(score_func=mutual_info_classif, k=mi_k)
    X_train_selected = mi_selector.fit_transform(
        X_train_encoded_df, y_train_resampled)
    selected_features = X_train_encoded_df.columns[mi_selector.get_support()]

    # Convert back to DataFrame after MI selection
    X_train_selected_df = pd.DataFrame(
        X_train_selected, columns=selected_features)

    # Print selected features count
    print(
        f"\nSelected {X_train_selected_df.shape[1]} features using Mutual Information:\n", selected_features.tolist())

    # Second feature selection step using model-based selection
    rf_selector = SelectFromModel(RandomForestClassifier(
        n_estimators=rf_n_estimators, random_state=random_state), threshold="median")
    X_train_final = rf_selector.fit_transform(
        X_train_selected_df, y_train_resampled)
    final_selected_features = X_train_selected_df.columns[rf_selector.get_support(
    )]

    # Convert back to DataFrame after RF selection
    X_train_final_df = pd.DataFrame(
        X_train_final, columns=final_selected_features)

    # Print final selected features count
    print(
        f"\nSelected {X_train_final_df.shape[1]} features after Random Forest selection:\n", final_selected_features.tolist())

    # Train final pipeline
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('mi_selection', mi_selector),  # First round of feature selection
        ('rf_selection', rf_selector),  # Second round of feature selection
        ('classifier', RandomForestClassifier(n_estimators=rf_n_estimators,
         random_state=random_state))  # Placeholder model
    ])

    # Fit pipeline with balanced data
    pipeline.fit(X_train_resampled, y_train_resampled)

    # Save encoders and scalers for production use
    joblib.dump(pipeline.named_steps['preprocessing'], os.path.join(
        save_path, "preprocessor.pkl"))
    joblib.dump(pipeline.named_steps['mi_selection'], os.path.join(
        save_path, "mi_selector.pkl"))
    joblib.dump(pipeline.named_steps['rf_selection'], os.path.join(
        save_path, "rf_selector.pkl"))

    # Ensure the importance scores align with the selected features
    feature_importance_values = rf_selector.estimator_.feature_importances_[
        :len(final_selected_features)]

    # Create DataFrame with feature names and their importance scores
    feature_description = pd.DataFrame({
        'Feature': final_selected_features,
        'Importance': feature_importance_values
    }).sort_values(by='Importance', ascending=False)  # Sort by importance

    print("Pipeline training completed.")

    return pipeline, X_train, X_test, y_train, y_test, feature_description

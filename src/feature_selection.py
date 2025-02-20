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


from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import os

import os
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def identify_feature_types(df, target_column):
    """Identify categorical and numerical columns in the dataframe."""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
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
    categorical_cols, numerical_cols = identify_feature_types(df, target_column)
    preprocessor = create_preprocessing_pipeline(categorical_cols, numerical_cols)

    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Transform data
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    # Convert to DataFrame
    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=feature_names)
    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=feature_names)

    return X_train_encoded_df, X_test_encoded_df, y_train, y_test, preprocessor

def apply_smote(X_train, y_train, random_state=42):
    """Apply SMOTE to balance the dataset."""
    smote = SMOTE(sampling_strategy='auto', random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

def select_features_mutual_info(X_train, y_train, k=50):
    """Select the top K features based on Mutual Information."""
    k = min(k, X_train.shape[1])  # Ensure k doesn't exceed the number of features
    mi_selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_selected = mi_selector.fit_transform(X_train, y_train)
    selected_features = X_train.columns[mi_selector.get_support()]
    
    print(f"\nSelected {len(selected_features)} features using Mutual Information:\n", selected_features.tolist())
    
    return X_train_selected, selected_features, mi_selector

def select_features_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """Select features using a Random Forest model-based selection."""
    rf_selector = SelectFromModel(RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state), threshold="median")
    
    X_train_final = rf_selector.fit_transform(X_train, y_train)
    final_selected_features = X_train.columns[rf_selector.get_support()]
    
    print(f"\nSelected {len(final_selected_features)} features after Random Forest selection:\n", final_selected_features.tolist())
    
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
    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df, target_column, test_size, random_state)

    print("Applying SMOTE...")
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train, random_state)

    print("Selecting features using Mutual Information...")
    X_train_selected, selected_features, mi_selector = select_features_mutual_info(X_train_resampled, y_train_resampled, mi_k)

    print("Selecting features using Random Forest model...")
    X_train_final, final_selected_features, rf_selector = select_features_random_forest(
        pd.DataFrame(X_train_selected, columns=selected_features), y_train_resampled, rf_n_estimators, random_state
    )

    # Train final pipeline
    print("Training final pipeline...")
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('mi_selection', mi_selector),
        ('rf_selection', rf_selector),
        ('classifier', RandomForestClassifier(n_estimators=rf_n_estimators, random_state=random_state))
    ])

    pipeline.fit(X_train_resampled, y_train_resampled)

    # Save pipeline
    print("Saving pipeline and feature selection models...")
    save_pipeline(pipeline, preprocessor, mi_selector, rf_selector, save_path)

    # Create feature importance DataFrame
    feature_importance_values = rf_selector.estimator_.feature_importances_[:len(final_selected_features)]
    feature_description = pd.DataFrame({
        'Feature': final_selected_features,
        'Importance': feature_importance_values
    }).sort_values(by='Importance', ascending=False)

    print("Pipeline training completed.")
    return pipeline, X_train, X_test, y_train, y_test, feature_description



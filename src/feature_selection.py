import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = feature_importances.nlargest(n_features).index.tolist()

    return top_features


from sklearn.feature_selection import mutual_info_classif

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

import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def create_feature_selection_pipeline(df, target_column, mi_k=100, rf_n_estimators=100, test_size=0.2, random_state=42, save_path="model_assets"):
    """
    Creates a feature selection pipeline that:
    1. Handles missing values by imputing categorical and numerical features.
    2. Encodes categorical features using one-hot encoding.
    3. Scales numerical features using standardization.
    4. Performs feature selection using Mutual Information and Random Forest-based selection.
    5. Trains a Random Forest classifier as a placeholder model.
    6. Returns selected feature descriptions and saves encoders/scalers for production deserialization.
    
    Parameters:
    - df (pd.DataFrame): Input dataset.
    - target_column (str): Name of the target column.
    - mi_k (int): Number of top features to keep after Mutual Information selection. Default is 100.
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
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Ensure target column is not in feature list
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    
    # Handle missing values by adding a new category for categorical features
    categorical_imputer = SimpleImputer(strategy="constant", fill_value="Missing")
    numerical_imputer = SimpleImputer(strategy="mean")
    
    # One-hot encode categorical features
    categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    
    # Standardize numerical features
    numerical_scaler = StandardScaler()
    
    # First feature selection step using mutual information
    mi_selector = SelectKBest(score_func=mutual_info_classif, k=mi_k)
    
    # Second feature selection step using model-based selection
    rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=rf_n_estimators, random_state=random_state))
    
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
    
    # Full pipeline with hybrid feature selection
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('mi_selection', mi_selector),  # First round of feature selection
        ('rf_selection', rf_selector),  # Second round of feature selection
        ('classifier', RandomForestClassifier(n_estimators=rf_n_estimators, random_state=random_state))  # Placeholder model
    ])
    
    # Split dataset
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Fit pipeline
    pipeline.fit(X_train, y_train)
    
    # Save encoders and scalers for production use
    joblib.dump(pipeline.named_steps['preprocessing'], os.path.join(save_path, "preprocessor.pkl"))
    joblib.dump(pipeline.named_steps['mi_selection'], os.path.join(save_path, "mi_selector.pkl"))
    joblib.dump(pipeline.named_steps['rf_selection'], os.path.join(save_path, "rf_selector.pkl"))
    
    # Get selected features
    selected_features = np.array(X_train.columns)[pipeline.named_steps['rf_selection'].get_support()]
    
    # Create feature description dataframe
    feature_description = pd.DataFrame({
        'Feature': selected_features,
        'Min': [df[feature].min() if feature in df else None for feature in selected_features],
        'Max': [df[feature].max() if feature in df else None for feature in selected_features],
        'Unique_Values': [df[feature].nunique() if feature in df else None for feature in selected_features]
    })
    
    print("Selected features:", selected_features)
    
    return pipeline, X_train, X_test, y_train, y_test, feature_description

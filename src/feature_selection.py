from sklearn.ensemble import RandomForestClassifier
import pandas as pd

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

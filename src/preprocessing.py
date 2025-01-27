import os
import pandas as pd

from src.config import DATASET_ROOT_PATH

def extract_wave_data(df, wave_number):
    """
    Extract variables from a specific wave based on the second character of their names.
    Section 1.4 of the harmonized MHAS documentation (version C.2) explains that the second character
    of any variable refers to the particular wave the variable is encoding.

    Parameters:
    - df: DataFrame
    - wave_number: int or str, wave identifier (1, 2, 3, 4, 5; the character 'a' denotes a cross-wave variable)

    Returns:
    - DataFrame with columns for the specified wave.
    """
    # Select variables from a specific wave
    specific_wave_columns = [col for col in df.columns if len(
        col) > 1 and col[1] == str(wave_number)]

    # Select cross-wave variables
    cross_wave = [col for col in df.columns if len(col) > 1 and col[1] == 'a']

    # Combine specific and common variables
    wave_columns = list(set(specific_wave_columns + cross_wave))

    return df[wave_columns]


def extract_respondent_data(df):
    """
    Extract variables from respondent.
    Section 1.4 of the harmonized MHAS documentation (version C.2) explains that the first character
    of any variable refers to the particular individual referred to by the variable.

    Parameters:
    - df: DataFrame    

    Returns:
    - DataFrame with columns for respondent.
    """
    return df[[col for col in df.columns if col.startswith('r')]]


def remove_missing_values(df, column_name):
    """
    Remove all rows matching missing values from a specified column

    Parameters:
    - df: DataFrame
    - column_name: name of column to search for missing values    

    Returns:
    - DataFrame with no missing values for the specified column.
    """

    df = df[df[column_name].notna()]

    return df


def missing_value_ratio(df, ratio):
    """
    Identify variables with the specified missing value ratio

    Parameters:
    - df: DataFrame
    - ratio: proportion of missing values    

    Returns:
    - List of columns with a ratio equal to or higher than the one specified by the user.
    """

    # Identify categorical columns
    # categorical_columns = df.select_dtypes(
    # include=['object', 'category']).columns

    # Identify and store columns with the specified missing value ratio
    columns_matching_missing_value_ratio = [
        col for col in df.columns
        if df[col].isnull().mean() > ratio
    ]

    print(f"Variables with a missing value ratio higher than {ratio}: {columns_matching_missing_value_ratio}")
    print(f"Count of variables with a missing ratio higher than {ratio}: {len(columns_matching_missing_value_ratio)}")

    return columns_matching_missing_value_ratio




def save_categorical_features_with_values(df: pd.DataFrame, file_name: str):
    """
    Saves the unique values of categorical features in a DataFrame to a text file.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        file_name (str): The name of the text file to save the output.
    """
    try:
        # Select only categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        file_path = os.path.join(DATASET_ROOT_PATH, file_name)
        # Open the file for writing
        with open(file_path, 'w') as f:
            for column in df.columns:
                # Get the unique values of the column
                unique_values = df[column].unique()
                # Limit the number of values shown for readability
                unique_values_preview = unique_values[:10]  # Show up to 10 values
                # Write the feature name and unique values to the file
                f.write(f"{column}: {list(unique_values_preview)}\n\n")

        print(f"Categorical features with their unique values have been saved to '{file_name}'")
    except Exception as e:
        print(f"An error occurred: {e}")

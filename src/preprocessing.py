import matplotlib.pyplot as plt
import os
import pandas as pd
import re

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

    print(
        f"Variables with a missing value ratio higher than {ratio}: {columns_matching_missing_value_ratio}")
    print(
        f"Count of variables with a missing ratio higher than {ratio}: {len(columns_matching_missing_value_ratio)}")

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
                # Show up to 10 values
                unique_values_preview = unique_values[:10]
                # Write the feature name and unique values to the file
                f.write(f"{column}: {list(unique_values_preview)}\n\n")

        print(
            f"Categorical features with their unique values have been saved to '{file_name}'")
    except Exception as e:
        print(f"An error occurred: {e}")


"""
Preprocess and stack dataset into a long format.

This function takes a dataset with multiple columns where the column names start with a letter followed by a number
(e.g., 'r1', 's2', 'h3', etc.). It transforms these columns into a long format, creating a new column that retains
only the postfix (removing the letter and number prefix). The original prefixed columns are deleted after transformation.

Parameters:
    file_path (str): Path to the input CSV file.
    output_path (str, optional): Path to save the transformed dataset. Defaults to 'transformed_data.csv'.

Returns:
    pd.DataFrame: Transformed dataframe in long format.
"""


def preprocess_and_stack_data(file_path, output_path="transformed_data.csv"):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Identify columns that start with any letter followed by a number
    pattern = re.compile(r'^[a-zA-Z]\d+')
    value_columns = [col for col in df.columns if pattern.match(col)]

    # Preserve relevant columns (excluding the identified ones)
    id_vars = [col for col in df.columns if col not in value_columns]

    # Melt the dataframe into long format
    df_long = df.melt(id_vars=id_vars, value_vars=value_columns,
                      var_name='original_column', value_name='value')

    # Extract the postfix (removing the prefix letter and number)
    df_long['postfix'] = df_long['original_column'].str.extract(
        r'^[a-zA-Z](\d+)(.*)')[1]

    # Drop the original column name column
    df_long.drop(columns=['original_column'], inplace=True)

    # Save the transformed dataset
    df_long.to_csv(output_path, index=False)

    print(f"Transformation complete. Saved as '{output_path}'")

    return df_long


# Auxiliary Functions


def extract_respondent_data(df):
    """
    Extract variables from respondent.
    Section 1.4 of the harmonized MHAS documentation (version C.2) explains that the first character
    of any variable refers to the particular individual referred to by the variable.
    """
    return df[[col for col in df.columns if col.startswith('r')]]


def filter_by_age(df, age_column='agey', age_threshold=50):
    """
    Filter rows by age threshold.

    Args:
        df (pd.DataFrame): The input dataset.
        age_column (str): The column containing age data.
        age_threshold (int): The minimum age to include.

    Returns:
        pd.DataFrame: Filtered dataset.
    """
    return df[df[age_column] >= age_threshold]


def filter_alive_respondents(df, status_column='iwstat', alive_value='1.Resp, alive'):
    """
    Filter the dataset to include only alive respondents based on the R5IWSTAT variable.

    Args:
        df (pd.DataFrame): Input dataset.
        status_column (str): Column name indicating respondent's status.
        alive_value (str): Value in the status column that represents 'alive'.

    Returns:
        pd.DataFrame: Filtered dataset containing only alive respondents.
    """
    return df[df[status_column] == alive_value]


def remove_columns_with_missing_values(df, missing_values_threshold=30, visualize=True):
    """
    Analyze missing values, optionally visualize them, and drop columns exceeding the threshold.

    Args:
        df (pd.DataFrame): Input dataframe.
        missing_values_threshold (int): Percentage threshold for dropping columns.
        visualize (bool): Whether to visualize the missing data.

    Returns:
        pd.DataFrame: Dataframe after dropping columns with excessive missing values.
    """
    # Calculate the percentage of missing values
    missing_percentage = (df.isnull().sum() / len(df)) * 100

    if visualize:
        # Visualize missing data
        plt.figure(figsize=(10, 6))
        colors = [
            'red' if val > missing_values_threshold else 'skyblue' for val in missing_percentage]
        missing_percentage.sort_values(
            ascending=False).plot(kind='bar', color=colors)
        plt.axhline(y=missing_values_threshold, color='gray', linestyle='--',
                    linewidth=1.5, label=f'Threshold ({missing_values_threshold}%)')
        plt.title("Percentage of Missing Values by Column", fontsize=16)
        plt.xlabel("Columns", fontsize=12)
        plt.ylabel("Percentage of Missing Values", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.show()

    # Identify and drop columns exceeding the threshold
    columns_to_drop = missing_percentage[missing_percentage >
                                         missing_values_threshold].index
    print(
        f"Dropping {len(columns_to_drop)} columns (>{missing_values_threshold}% missing): {list(columns_to_drop)}")

    # Drop the columns
    return df.drop(columns=columns_to_drop)


def process_target_column(df, target_column='hosp1y'):
    """
    Process the target column by removing missing values and mapping categorical values to numerical.

    Args:
        df (pd.DataFrame): Input dataframe.
        target_column (str): Target column to process.

    Returns:
        pd.DataFrame: Dataframe after processing the target column.
    """
    target_mapping = {'1.Yes': 1, '0.No': 0}

    if target_column in df.columns:
        # Remove rows with missing target values
        df = df.dropna(subset=[target_column])
        df[target_column] = df[target_column].map(target_mapping)

    return df


def run_preprocessing_pipeline(df, missing_percentage_threshold=70, target_column='hosp1y'):
    """
    Runs the entire preprocessing pipeline on the input dataset.

    Args:
        df (pd.DataFrame): The raw input dataset.
        missing_percentage_threshold (int): Threshold for dropping columns with missing values.
        target_column (str): The target column for prediction.

    Returns:
        pd.DataFrame: Preprocessed and cleaned dataset.
    """
    print("Starting data preprocessing pipeline...\n")

    # Step 1: Include only direct respondents
    # print("Filtering data for direct respondents...")
    # df = extract_respondent_data(df)
    # print(f"Shape after filtering respondents: {df.shape}")

    # Step 2: Filter for elderly individuals aged 50 or older
    print("Filtering data for elderly individuals (aged 50+)...")
    df = filter_by_age(df)
    print(f"Shape after filtering by age: {df.shape}")

    # Step 3: Exclude individuals who passed away
    print("Excluding individuals who passed away during the waves...")
    df = filter_alive_respondents(df)
    print(f"Shape after filtering alive respondents: {df.shape}")

    # Step 4: Remove columns with too many missing values
    print(
        f"Removing columns with more than {missing_percentage_threshold}% missing values...")
    df = remove_columns_with_missing_values(df, missing_percentage_threshold)
    print(f"Shape after removing columns: {df.shape}")

    # Step 5: Process the target column
    print(f"Processing the target column '{target_column}'...")
    df = process_target_column(df, target_column)
    print(f"Shape after processing target column: {df.shape}")

    return df


"""
Preprocess and stack dataset.

This function filters all columns that start with "r" followed by a number from 1 to 5,
and then creates separate dataframes per wave. Each dataframe is merged into a final dataframe
where wave-specific columns (e.g., r1meal, r2meal, r3meal) are consolidated into a single column
without the prefix.

Parameters:
    file_path (str): Path to the input CSV file.
    output_path (str, optional): Path to save the transformed dataset. Defaults to 'stacked_data.csv'.

Returns:
    pd.DataFrame: Stacked dataframe.
"""


def stack_all_waves_respondent_data(file_path, output_path="stacked_data.csv"):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Identify columns that start with 'r' followed by a number from 1 to 5
    pattern = re.compile(r'^r[1-5]')
    r_columns = [col for col in df.columns if pattern.match(col)]

    # Identify general columns to preserve
    id_vars = [col for col in df.columns if col not in r_columns]

    # Create separate dataframes for each wave
    wave_dfs = []
    for wave in range(1, 6):
        wave_pattern = re.compile(f'^r{wave}')
        wave_cols = [col for col in df.columns if wave_pattern.match(col)]

        # Rename wave-specific columns to remove prefix
        df_wave = df[wave_cols].copy()
        df_wave.rename(columns=lambda x: re.sub(
            f'^r{wave}', '', x) if x in wave_cols else x, inplace=True)
        df_wave['wave'] = wave  # Add wave identifier
        wave_dfs.append(df_wave)

    # Concatenate all wave dataframes
    df_stacked = pd.concat(wave_dfs, ignore_index=True)

    # Save the transformed dataset
    df_stacked.to_csv(output_path, index=False)

    print(f"Transformation complete. Saved as '{output_path}'")

    return df_stacked

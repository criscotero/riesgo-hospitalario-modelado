import gc

import pandas as pd


def join_prefix_suffix(waves: range, suffix: str,) -> set:

    prefixes = {f"r{wave}" for wave in waves} | {f"s{wave}" for wave in waves}

    joined_variables = {f"{prefix}{suffix}" for prefix in prefixes}

    return joined_variables


def stack_df(df: pd.DataFrame, target_variables: set, selected_features_suffixes: list, rural) -> pd.DataFrame:

    # selected_features_columns = {
    #    f"{prefix}{feature}" for feature in selected_features_suffixes for prefix in {'r1', 'r2', 's1', 's2'}
    # }
    #
    # <--- VERIFY COLUMNS HAVE BEEN PROPERLY GENERATED --->
    # if (len(selected_features_columns)) == 224:
    #    print(f"All columns add up! :)")
    # else:
    #    print(f"Columns are not adding up! :(")

    # <---- BEGIN ITERATION OF EACH TARGET FOR EACH ROW ---->
    new_target = 'hospitalized'
    new_rows = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict()

        for target in target_variables:
            wave = int(target[1])
            prefix = target[:2]

            new_row = {'id': row_dict['unhhidnp'], 'wave': wave}

            hospitalization_value = row_dict.get(target, None)
            new_row[new_target] = hospitalization_value

            gender_key = 'ragender' if prefix.startswith(
                'r') else f's{wave}gender'
            new_row['gender'] = row_dict.get(gender_key, None)

            if rural == True:
                hWrural_col = f'h{wave}rural'
                new_row['location'] = row_dict[hWrural_col]

            for feature in selected_features_suffixes:
                col_name = f'{prefix}{feature}'
                if col_name in row_dict:
                    new_row[feature] = row_dict[col_name]

            new_rows.append(new_row)

    stacked_df = pd.DataFrame(new_rows)
    return stacked_df


def find_non_cross_wave_columns(columns: list, waves: set[str]) -> dict:
    """
    Finds columns that are not present in all 5 waves.
    """

    suffix_to_waves = {}

    for col in columns:
        if col[0] not in {'r', 's', 'h'} or col[1] not in waves:
            continue

        wave = col[1]  # the second character is the wave number
        suffix = col[2:]  # the rest of the column name is the suffix
        suffix_to_waves.setdefault(suffix, set()).add(wave)

    missing = {}
    for suffix, waves_found in suffix_to_waves.items():
        missing_waves = waves - waves_found
        if missing_waves:
            missing[suffix] = missing_waves
    return missing


def generate_waved_columns(wave_range: tuple, column_suffix: list) -> list:
    """
    Dynamically generates waved columns from a list of column suffixes.

    Args:
        column_suffix (list): The list of column suffixes to be prefixed with waves.

    Returns:
        list: list with generated waved columns.
    """

    return [
        f'r{wave}{suffix}'
        for wave in range(wave_range[0], wave_range[1] + 1)
        for suffix in column_suffix
    ]


def categorize_columns(df: pd.DataFrame) -> dict:
    """
    Categorizes columns into those that start with a specified prefix 
    and those that do not match this pattern.
    """

    waves = {str(i) for i in range(1, 6)} | {'a'}

    rX_prefix = {f'r{wave}' for wave in waves}
    sX_prefix = {f's{wave}' for wave in waves}
    hX_prefix = {f'h{wave}' for wave in waves}

    rX_columns = [
        col for col in df.columns if col.startswith(tuple(rX_prefix))]
    sX_columns = [
        col for col in df.columns if col.startswith(tuple(sX_prefix))]
    hX_columns = [
        col for col in df.columns if col.startswith(tuple(hX_prefix))]

    sX_hX_columns = set(sX_columns) | set(hX_columns)

    non_waved_columns = [col for col in df.columns if col not in (
        rX_columns + list(sX_hX_columns))]

    return {'rX_columns': rX_columns, 'hX_columns': hX_columns, 'sX_columns': sX_columns, 'non_waved_columns': non_waved_columns}


def compute_proportions(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Computes the count and proportion of 1s and 0s for a given list of columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the columns.
        columns (list): The list of column names to analyze.

    Returns:
        pd.DataFrame: A DataFrame with counts and proportions of 1s and 0s.
    """

    return df[columns].apply(lambda col: col.value_counts(normalize=True, dropna=False))

import pandas as pd
def unified_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines spouse (s) and respondent (r) data for waves 3, 4, and 5 into a single DataFrame,
    excluding columns related to the householder (h).

    Args:
        df (pd.DataFrame): Input DataFrame containing data for respondents, spouses, and householders.

    Returns:
        pd.DataFrame: Combined DataFrame with data for respondents and spouses for waves 3, 4, and 5.
    """
    all_columns = df.columns
    combined_df = pd.DataFrame()

    # Process selected waves (3, 4, 5)
    for wave in range(3, 6):
        # Select respondent columns for the current wave, excluding householder columns
        respondent_cols = [col for col in all_columns if col.startswith(f"r{wave}") and not col.startswith("h")]
        spouse_cols = [col.replace(f"r{wave}", f"s{wave}", 1) for col in respondent_cols]

        # Filter spouse columns to only include those that exist in the DataFrame
        spouse_cols = [col for col in spouse_cols if col in all_columns]

        # Add respondent data for the wave
        df_r_wave = df[respondent_cols].copy()
        df_r_wave.columns = ["m" + col[2:] for col in respondent_cols]

        # Add spouse data for the wave
        df_s_wave = pd.DataFrame()
        if spouse_cols:  # Only process if there are valid spouse columns
            df_s_wave = df[spouse_cols].copy()
            df_s_wave.columns = ["m" + col[2:] for col in respondent_cols if col.replace(f"r{wave}", f"s{wave}", 1) in spouse_cols]

        # Concatenate the data for the current wave
        wave_combined = pd.concat([df_r_wave, df_s_wave], ignore_index=True)

        # Append to the final combined DataFrame
        combined_df = pd.concat([combined_df, wave_combined], ignore_index=True)

    return combined_df

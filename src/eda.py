import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def plot_missing_values(dataframe:pd.DataFrame, sorted=True ,display_xticklabels=False):
    """_summary_

    Args:
        dataframe (pd.DataFrame): _description_
        sorted (bool, optional): _description_. Defaults to True.
        display_xticklabels (bool, optional): _description_. Defaults to False.
    """

    # Calculate the percentage of missing values
    missing_values_series = dataframe.isna().mean() * 100
    df = pd.DataFrame(missing_values_series, columns=["Percentage of Missings"])
    if sorted:
        df = df.sort_values(by="Percentage of Missings", ascending=False)

    with plt.style.context('bmh'):
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=df.index, y="Percentage of Missings", data=df, linewidth=0)

        if display_xticklabels:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        else:
            ax.set_xticklabels([])

        ax.set_xlabel("Feature")
        ax.set_ylabel("Percentage of Missings")
        ax.set_title("Data Missing Values")

        # Adjust the layout to prevent label overlap
        plt.tight_layout()
        plt.show()




def plot_missing_percentage(df: pd.DataFrame, column: str):
    """
    Plots the percentage of missing values in the specified column of the dataframe.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        column (str): The column name to analyze for missing values.
    """
    total_rows = len(df)
    missing_count = df[column].isna().sum()
    missing_percentage = (missing_count / total_rows) * 100

    # Create DataFrame for plotting
    data = pd.DataFrame({'Column': [column], 'Missing Percentage': [missing_percentage]})

    # Plot
    plt.figure(figsize=(6, 4))
    sns.barplot(x='Column', y='Missing Percentage', data=data, palette='Set2')

    # Labels & Title
    plt.ylim(0, 100)
    plt.ylabel('Missing Values (%)')
    plt.title(f'Missing Values in {column}')

    # Display percentage on the bar
    for index, value in enumerate(data['Missing Percentage']):
        plt.text(index, value + 2, f'{value:.2f}%', ha='center', fontsize=12)

    plt.show()


def plot_age_histogram(df: pd.DataFrame, age_column: str = 'r5agey', bin_width: int = 10):
    """
    Plots a histogram of the age distribution in bins of a given width.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        age_column (str): The column name that represents age.
        bin_width (int): The width of each age bin (default is 10 years).
    """
    if age_column not in df.columns:
        raise ValueError(f"Column '{age_column}' not found in the DataFrame")

    plt.figure(figsize=(10, 5))
    sns.histplot(df[age_column], bins=range(30, 100, bin_width), kde=False, color="royalblue")

    # Labels and title
    plt.xlabel("Age Groups")
    plt.ylabel("Count of Participants")
    plt.title(f"Age Distribution of Participants ({age_column})")
    
    # Set x-ticks to match bin edges
    plt.xticks(range(30, 100, bin_width))

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
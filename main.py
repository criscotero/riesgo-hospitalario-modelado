from src.extract_data import get_datasets


def main():
     try:
        dataframe = get_datasets(chunksize=10000)
        print("Dataset loaded successfully!")
        print(dataframe.head())  # Display the first few rows
     except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
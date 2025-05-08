from toolkit import DataParser
import numpy as np
import pandas as pd
import sys

def split_dataset(data):
    '''
    Loads a dataset, replaces NaN values, and splits it into training and validation sets.

    Parameters:
    data (str): Path to the dataset file (CSV format).

    Returns:
        pd.DataFrame: Training set (80% of the data).
        pd.DataFrame: Validation set (20% of the data).
    '''
    data = DataParser.replace_nan_values(DataParser.open_file(data, None))

    np.random.seed(42)
    train_ratio = 0.8
    index = np.arange(data.shape[0])
    np.random.shuffle(index)
    train_size = int(data.shape[0] * train_ratio)

    train_index = index[:train_size]
    valid_index = index[train_size:]
    train_set = data.iloc[train_index]
    valid_set = data.iloc[valid_index]

    print(f"X_train shape : {train_set.iloc[:,2:].shape}")
    print(f"X_valid shape : {valid_set.iloc[:,2:].shape}")

    return train_set, valid_set

def main():
    '''
    Entry point of the script. Validates command-line arguments,
    splits the dataset into training and validation sets,
    and saves them as CSV files.

    Usage:
    python3 ./split_data.py dataset_name

    Output:
    - train.csv: File containing the training data.
    - validation.csv: File containing the validation data.
    '''
    if (len(sys.argv) != 2):
        print("Usage: python3 ./split_data.py dataset_name")
        sys.exit(1)

    train, test = split_dataset(sys.argv[1])
    train.to_csv("train.csv", index=False, header=False)
    test.to_csv("validation.csv", index=False, header=False)

if __name__ == "__main__":
    main()
from toolkit import DataParser
import numpy as np
import pandas as pd
import sys

def split_dataset(data):
    data = DataParser.replace_nan_values(DataParser.open_file(data, None))

    np.random.seed(42)
    train_ratio = 0.8
    index = np.arange(data.shape[0])
    np.random.shuffle(index)
    train_size = int(data.shape[0] * train_ratio)

    train_index = index[:train_size]
    valid_index = index[train_size:]
    train_set = data.iloc[train_index, 2:]
    valid_set = data.iloc[valid_index, 2:]

    print(f"x_train shape: {train_set.shape}")
    print(f"x_valid shape: {valid_set.shape}")

    return train_set, valid_set

def main():
    if (len(sys.argv) != 2):
        print("Usage: python3 ./split_data.py dataset_name")
        sys.exit(1)
    train, test = split_dataset(sys.argv[1])
    train.to_csv("train.csv", index=False)
    test.to_csv("validation.csv", index=False)

if __name__ == "__main__":
    main()
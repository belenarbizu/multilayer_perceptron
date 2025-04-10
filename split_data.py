from toolkit import DataParser
import numpy as np
import pandas as pd
import sys

def split_dataset(data):
    data = DataParser.replace_nan_values(DataParser.open_file(data, None))
    print(data.head())
    np.random.seed(42)
    train_ratio = 0.8
    index = np.arange(data.shape[0])
    np.random.shuffle(index)
    train_size = int(data.shape[0] * train_ratio)

    train_index = index[:train_size]
    test_index = index[train_size:]
    train_set = data.iloc[train_index]
    test_set = data.iloc[test_index]

    return train_set, test_set

def main():
    if (len(sys.argv) != 2):
        print("Usage: python3 ./split_data.py dataset_name")
        sys.exit(1)
    train, test = split_dataset(sys.argv[1])
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)

if __name__ == "__main__":
    main()
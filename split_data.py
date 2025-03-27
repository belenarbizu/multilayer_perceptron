from toolkit import DataParser
import numpy as np
import pandas as pd
import sys

def split_dataset(data):
    data = DataParser.replace_nan_values(DataParser.open_file(data))
    train_set = 0 # 80%
    test_set = 0 # 20%
    return train_set, test_set

def main():
    if (len(sys.argv) != 2):
        print("Usage: python3 ./train.py dataset_name")
        sys.exit(1)
    train, test = split_dataset(sys.argv[1])
    
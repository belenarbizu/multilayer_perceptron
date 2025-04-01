from toolkit import DataParser
import sys
import numpy as np
import argparse

class Network():

    def __init__(self, file, vars):
        self.data = DataParser.replace_nan_values(DataParser.open_file(file, 0))
        vars_list = ["layer", "epochs", "loss", "batch_size", "learning_rate"]
        for var, name in zip(vars, vars_list):
            setattr(self, name, vars[name])
        self.weights = {}
        self.bias = {}
        self.mean = {}
        self.std = {}


    def standardize(self):
        for col in self.data.columns:
            if (self.data[col].dtype == "int64" or self.data[col].dtype == "float64"):
                self.mean[col] = np.mean(self.data[col])
                self.std[col] = np.std(self.data[col])
                self.data[col] = (self.data[col] - self.mean[col]) / self.std[col]


def main():
    parser = argparse.ArgumentParser(description='Predicts whether a cancer is malignant or benign')
    parser.add_argument('-l', '--layer', nargs='+', help='Layers of model')
    parser.add_argument('-e', '--epochs', help='Number of epochs')
    parser.add_argument('-s', '--loss', help='Loss function')
    parser.add_argument('-b', '--batch_size', help='Size of the batch')
    parser.add_argument('-r', '--learning_rate', help='Learning rate')
    args = parser.parse_args()

    nn = Network("train.csv", vars(args))
    nn.standardize()

if __name__ == "__main__":
    main()
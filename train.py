from toolkit import DataParser
import sys
import numpy as np

class Network():

    def __init__(self, file):
        self.data = DataParser.replace_nan_values(DataParser.open_file(file, 0))
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


    def weighted_sum(self):
        sum = 0
        return sum

def main():
    if (len(sys.argv) != 2):
        print("Usage: python3 ./train.py dataset_name")
        sys.exit(1)
    nn = Network(sys.argv[1])
    nn.standardize()

if __name__ == "__main__":
    main()
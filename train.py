from toolkit import DataParser
from layer import Layer
import sys
import numpy as np
import argparse

class Network:

    def __init__(self, train_file, test_file, vars):
        self.train_data = DataParser.replace_nan_values(DataParser.open_file(train_file, 0, 0))
        self.test_data = DataParser.replace_nan_values(DataParser.open_file(test_file, 0, 0))

        vars_list = ["layer", "epochs", "loss", "batch_size", "learning_rate"]
        for var, name in zip(vars, vars_list):
            setattr(self, name, vars[name])

        self.categories = ["M", "B"]
        labels_train = {}
        labels_test = {}
        for cat in self.categories:
            labels_train[cat] = (self.train_data.iloc[:,0] == cat).astype(int)
            labels_test[cat] = (self.test_data.iloc[:,0] == cat).astype(int)

        for cat, label in labels_train.items():
            self.train_data[f"{cat}_label"] = label
        
        for cat, label in labels_test.items():
            self.test_data[f"{cat}_label"] = label

        self.train_data = self.train_data.iloc[:,1:]
        self.test_data = self.test_data.iloc[:,1:]

        self.n_inputs = self.train_data.iloc[:,:-2].shape[1]
        self.n_layers = len(self.layer)
        if self.n_layers == 1:
            self.n_layers = 2
            self.layer.append(self.layer[0])

        self.mean = {}
        self.std = {}


    def standardize(self):
        for col in self.train_data.columns:
            if ((self.train_data[col].dtype == "int64" or self.train_data[col].dtype == "float64") and col != "M_label" and col != "B_label"):
                self.mean[col] = np.mean(self.train_data[col])
                self.std[col] = np.std(self.train_data[col])
                self.train_data[col] = (self.train_data[col] - self.mean[col]) / self.std[col]
        for col in self.test_data.columns:
            if ((self.test_data[col].dtype == "int64" or self.test_data[col].dtype == "float64") and col != "M_label" and col != "B_label"):
                self.test_data[col] = (self.test_data[col] - self.mean[col]) / self.std[col]


    def create_layers(self):
        self.layers = []
        self.train_data_training = self.train_data.iloc[:,:-2]

        input_layer = Layer(self.n_inputs, self.layer[0])
        self.layers.append(input_layer)
    
        for num in range(self.n_layers):
            if num == self.n_layers - 1:
                layer = Layer(self.layer[num], self.layer[num])
            else:
                layer = Layer(self.layer[num], self.layer[num + 1])
            self.layers.append(layer)

        output_layer = Layer(self.layer[self.n_layers - 1], 2)
        self.layers.append(output_layer)


    def train(self):
        x = self.train_data_training.values
        y = self.train_data[["M_label", "B_label"]].values
        x_val = self.test_data.iloc[:,:-2].values
        y_val = self.test_data[["M_label", "B_label"]].values

        for epoch in range(self.epochs):
            output = x
            for layer in range(len(self.layers) - 1):
                output = self.layers[layer].forward(output)

            out = self.layers[-1].forward(output)
            y_pred = DataParser.softmax(out)
            loss = self.categorical_cross_entropy(y, y_pred)

            dinputs = y_pred - y
            for i in reversed(range(len(self.layers))):
                layer = self.layers[i]
                if i != len(self.layers) - 1 and i != 0:
                    dinputs *= DataParser.relu_derivative(layer.output)
                dinputs = layer.backward(dinputs)
            
            for layer in self.layers:
                layer.weights -= self.learning_rate * layer.dweights
                layer.bias -= self.learning_rate * layer.dbiases

            val_output = x_val
            for layer in range(len(self.layers) - 1):
                val_output = self.layers[layer].forward(val_output)
            out_val = self.layers[-1].forward(val_output)
            y_val_pred = DataParser.softmax(out_val)
            val_loss = self.categorical_cross_entropy(y_val, y_val_pred)

            if epoch % 10 == 0:
                self.print_info(epoch, loss, val_loss)


    def categorical_cross_entropy(self, true_values, predicted_values):
        epsilon = 1e-15
        loss = - np.sum(true_values * np.log(predicted_values + epsilon), axis=1)
        return np.mean(loss)


    def print_info(self, epoch, loss, val_loss):
        print(f"epoch {epoch}/{self.epochs} - loss: {loss} - val_loss: {val_loss}")


def main():
    parser = argparse.ArgumentParser(description='Predicts whether a cancer is malignant or benign')
    parser.add_argument('-l', '--layer', nargs='+', type=int, default=[24, 24] ,help='Layers of model')
    parser.add_argument('-e', '--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('-s', '--loss', default='categoricalCrossEntropy', help='Loss function')
    parser.add_argument('-b', '--batch_size', type=int, default=None, help='Size of the batch')
    parser.add_argument('-r', '--learning_rate', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    nn = Network("train.csv", "validation.csv", vars(args))
    nn.standardize()
    nn.create_layers()
    nn.train()

if __name__ == "__main__":
    main()
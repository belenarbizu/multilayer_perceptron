from toolkit import DataParser
from layer import Layer
import sys
import numpy as np
import argparse

class Network():

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
            labels_train[cat] = (self.train_data.iloc[:,0:1] == cat).astype(int)
            labels_test[cat] = (self.test_data.iloc[:,0:1] == cat).astype(int)

        for cat, label in labels_train.items():
            self.train_data[f"{cat}_label"] = label
        
        for cat, label in labels_test.items():
            self.test_data[f"{cat}_label"] = label

        self.n_inputs = self.train_data.iloc[:,2:-2].shape[1] 
        self.n_layers = len(self.layer)
        if len(self.layer) < 2:
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


    def create_layers(self):
        self.layers = []
        self.train_data_training = self.train_data.iloc[:,2:-2]
        input_layer = Layer(self.n_inputs, self.layer[0])
        self.layers.append(input_layer)
        weighted_sum = input_layer.forward(self.train_data_training)
        output = DataParser.relu(weighted_sum)
    
        for n in range(self.n_layers):
            if n == self.n_layers - 1:
                layer = Layer(self.layer[n], self.layer[n])
            else:
                layer = Layer(self.layer[n], self.layer[n + 1])
            self.layers.append(layer)
            weighted_sum = layer.forward(output)
            output = DataParser.relu(weighted_sum)

        output_layer = Layer(self.layer[self.n_layers - 1], 2)
        self.layers.append(output_layer)
        weighted_sum = output_layer.forward(output)
        output = DataParser.softmax(weighted_sum)
        #loss
        loss = self.categorical_cross_entropy(self.train_data.loc[:, ["M_label", "B_label"]], output)
        self.print_info(0, loss, 0)
        #val_loss
        #loss = self.categorical_cross_entropy(self.test_data.loc[:, ["M_label", "B_label"]], output)


    def categorical_cross_entropy(self, true_values, predicted_values):
        epsilon = 1e-15
        loss = - np.sum(true_values * np.log(predicted_values + epsilon), axis=1) / len(true_values)
        return np.mean(loss)


    def weights_gradient(self, x, y_label, y_predicted):
        m = len(y_label)
        grad = 1 / m * np.dot(x.T, (y_predicted - y_label))
        return grad
    
    
    def bias_gradient(self, y_label, y_predicted):
        m = len(y_label)
        grad = 1 / m * np.sum(y_predicted - y_label)
        return grad


    def print_info(self, epoch, loss, val_loss):
        print(f"epoch {epoch}/{self.epochs} - loss: {loss} - val_loss: {val_loss}")


def main():
    parser = argparse.ArgumentParser(description='Predicts whether a cancer is malignant or benign')
    parser.add_argument('-l', '--layer', nargs='+', type=int, default=[24, 24] ,help='Layers of model')
    parser.add_argument('-e', '--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('-s', '--loss', default='categoricalCrossEntropy', help='Loss function')
    parser.add_argument('-b', '--batch_size', type=int, default=None, help='Size of the batch')
    parser.add_argument('-r', '--learning_rate', type=float, default=0.1, help='Learning rate')
    args = parser.parse_args()

    nn = Network("train.csv", "test.csv", vars(args))
    nn.standardize()
    nn.create_layers()

if __name__ == "__main__":
    main()
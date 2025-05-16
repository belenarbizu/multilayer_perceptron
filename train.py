from toolkit import DataParser
from layer import Layer
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import copy

class Network:
    '''
    Implements a fully connected neural network for binary classification (malignant/benign).

    Attributes:
        train_data (DataFrame): Preprocessed training data.
        test_data (DataFrame): Preprocessed validation data.
        layer (list): List of hidden layer sizes.
        epochs (int): Number of training epochs.
        loss (str): Loss function used.
        batch_size (int): Batch size used during training.
        learning_rate (float): Learning rate for weight updates.
        categories (list): List of target categories ("M", "B").
        adam (bool): Flag to use Adam optimizer.
        layers (list): List of Layer objects composing the network.
        mean (dict): Mean values used for standardization.
        std (dict): Standard deviation values used for standardization.
        train_losses, val_losses, train_accuracies, val_accuracies (list): Training/validation metrics per epoch.
        early_stop (bool): Flag for early stopping.
    '''
    def __init__(self, train_file, test_file, vars):
        '''
        Initializes the neural network, loads and preprocesses data, and sets model configuration.

        Args:
            train_file (str): Path to the training CSV file.
            test_file (str): Path to the validation CSV file.
            vars (dict): Dictionary with model configuration.
        '''
        self.train_data = DataParser.replace_nan_values(DataParser.open_file(train_file, 0))
        self.test_data = DataParser.replace_nan_values(DataParser.open_file(test_file, 0))

        vars_list = ["layer", "epochs", "loss", "batch_size", "learning_rate", "adam"]
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
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.early_stop = False


    def standardize(self):
        '''
        Standardizes training and validation data using the training set's mean and standard deviation.
        '''
        for col in self.train_data.columns:
            if (self.train_data[col].dtype == "int64" or self.train_data[col].dtype == "float64") and not str(col).endswith('_label'):
                self.mean[col] = np.mean(self.train_data[col])
                self.std[col] = np.std(self.train_data[col])
                self.train_data[col] = (self.train_data[col] - self.mean[col]) / self.std[col]
        for col in self.test_data.columns:
            if (self.test_data[col].dtype == "int64" or self.test_data[col].dtype == "float64") and not str(col).endswith('_label'):
                self.test_data[col] = (self.test_data[col] - self.mean[col]) / self.std[col]


    def create_layers(self):
        '''
        Creates the layers of the neural network based on the specified architecture.
        '''
        self.layers = []
        self.train_data_training = self.train_data.iloc[:,:-2]

        input_layer = Layer(self.n_inputs, self.layer[0])
        self.layers.append(input_layer)
    
        for num in range(1, self.n_layers):
            layer = Layer(self.layer[num - 1], self.layer[num], activation='relu')
            self.layers.append(layer)

        output_layer = Layer(self.layer[self.n_layers - 1], 2, activation='softmax')
        self.layers.append(output_layer)

        if self.adam:
            for layer in self.layers:
                layer.m_w = np.zeros_like(layer.weights)
                layer.v_w = np.zeros_like(layer.weights)
                layer.m_b = np.zeros_like(layer.bias)
                layer.v_b = np.zeros_like(layer.bias)


    def train(self):
        '''
        Trains the neural network using backpropagation and updates weights based on the loss function.
        Tracks and stores loss and accuracy for each epoch.
        '''
        x = self.train_data_training.values
        y = self.train_data[["M_label", "B_label"]].values
        x_val = self.test_data.iloc[:,:-2].values
        y_val = self.test_data[["M_label", "B_label"]].values
        m = len(x)
        counter = 0
        best_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            for i in range(0, m, self.batch_size):
                x_batch = x[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                output = x_batch
                for layer in range(len(self.layers) - 1):
                    output = self.layers[layer].forward(output)

                out = self.layers[-1].forward(output)
                y_pred = DataParser.softmax(out)
                loss = self.categorical_cross_entropy(y_batch, y_pred)

                dinputs = y_pred - y_batch
                for i in reversed(range(len(self.layers))):
                    layer = self.layers[i]
                    if i != len(self.layers) - 1:
                        dinputs *= DataParser.relu_derivative(layer.output)
                    dinputs = layer.backward(dinputs)
                
                lambda_l2 = 0.001
                if self.adam:
                    self.adam_optimizer(lambda_l2, epoch)
                else:
                    for layer in self.layers:
                        layer.weights -= self.learning_rate * (layer.dweights + lambda_l2 * layer.weights)
                        layer.bias -= self.learning_rate * layer.dbiases

            self.save_loss(x, y)
            f1 = self.evaluate_f1(x, y)
            y_val_pred = self.validation_perf(x_val)

            val_loss = self.categorical_cross_entropy(y_val, y_val_pred)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(self.evaluate_accuracy(x_val, y_val))

            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0
                best_weights = [copy.deepcopy(layer.weights) for layer in self.layers]
                best_biases = [copy.deepcopy(layer.bias) for layer in self.layers]
            else:
                counter += 1

            if epoch == 1 or epoch % 10 == 0:
                self.print_info(epoch, loss, val_loss, f1)
            
            if counter >= 10:
                for i in range(len(self.layers)):
                    self.layers[i].weights = best_weights[i]
                    self.layers[i].bias = best_biases[i]
                self.early_stop = True
                print(f"Early stopping at epoch {epoch}")
                break


    def adam_optimizer(self, lambda_l2, epoch):
        '''
        Updates weights and biases using the Adam optimization algorithm with L2 regularization.

        Args:
            lambda_l2 (float): L2 regularization parameter.
            epoch (int): Current epoch.
        '''
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        t = epoch

        for layer in self.layers:
            layer.m_w = beta1 * layer.m_w + (1 - beta1) * (layer.dweights + lambda_l2 * layer.weights)
            layer.v_w = beta2 * layer.v_w + (1 - beta2) * ((layer.dweights + lambda_l2 * layer.weights) ** 2)

            m_w_hat = layer.m_w / (1 - beta1 ** t)
            v_w_hat = layer.v_w / (1 - beta2 ** t)

            layer.weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)

            layer.m_b = beta1 * layer.m_b + (1 - beta1) * layer.dbiases
            layer.v_b = beta2 * layer.v_b + (1 - beta2) * (layer.dbiases ** 2)

            m_b_hat = layer.m_b / (1 - beta1 ** t)
            v_b_hat = layer.v_b / (1 - beta2 ** t)

            layer.bias -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)


    def graphs(self):
        '''
        Plots two subplots:
            - Training and validation loss over epochs.
            - Training and validation accuracy over epochs.
            
        If early stopping was used, indicates the best epoch with a vertical red line.
        '''
        fig, axs = plt.subplots(1, 2, figsize=(25,13))
        epoch = range(0, len(self.train_losses))
        best_epoch = self.val_losses.index(min(self.val_losses))

        axs[0].plot(epoch, self.train_losses, label="training loss")
        axs[0].plot(epoch, self.val_losses, label="validation loss")
        if self.early_stop:
            axs[0].axvline(best_epoch, color='red', linestyle='--', label='Early stop')
        axs[0].set_xlabel("epochs")
        axs[0].set_ylabel("loss")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(epoch, self.train_accuracies, label="training acc")
        axs[1].plot(epoch, self.val_accuracies, label="validation acc")
        if self.early_stop:
            axs[1].axvline(best_epoch, color='red', linestyle='--', label='Early stop')
        axs[1].set_xlabel("epochs")
        axs[1].set_ylabel("accuracy")
        axs[1].legend()

        plt.tight_layout()
        plt.savefig("plot.png")
        plt.close(fig)


    def validation_perf(self, x_val):
        '''
        Evaluates the performance of the model on the validation set.

        Args:
            x_val (ndarray): Validation feature data.

        Returns:
            ndarray: Softmax predictions.
        '''
        val_output = x_val
        for layer in range(len(self.layers) - 1):
            val_output = self.layers[layer].forward(val_output)
        out_val = self.layers[-1].forward(val_output)
        y_val_pred = DataParser.softmax(out_val)
        return y_val_pred


    def save_loss(self, x, y):
        '''
        Saves the training loss and accuracy for the current epoch.

        Args:
            x (ndarray): Input features.
            y (ndarray): True labels.
        '''
        output = x
        for layer in range(len(self.layers) - 1):
            output = self.layers[layer].forward(output)
        out = self.layers[-1].forward(output)
        y_pred = DataParser.softmax(out)
        loss = self.categorical_cross_entropy(y, y_pred)
        accuracy = self.evaluate_accuracy(x, y)

        self.train_losses.append(loss)
        self.train_accuracies.append(accuracy)


    def categorical_cross_entropy(self, true_values, predicted_values):
        '''
        Computes the categorical cross-entropy loss.
        Args:
            true_values (ndarray): True labels.
            predicted_values (ndarray): Softmax predictions.

        Returns:
            float: Loss value.
        '''
        epsilon = 1e-15
        loss = - np.sum(true_values * np.log(predicted_values + epsilon), axis=1)
        return np.mean(loss)


    def print_info(self, epoch, loss, val_loss, f1):
        '''
        Prints the training and validation loss for the current epoch.
        
        Args:
            epoch (int): Current epoch number.
            loss (float): Training loss for the current epoch.
            val_loss (float): Validation loss for the current epoch.
        '''
        print(f"epoch {epoch}/{self.epochs} - loss: {loss} - val_loss: {val_loss} - f1: {f1}")


    def evaluate_accuracy(self, x, y):
        '''
        Evaluates the accuracy of the model on the given data.

        Args:
            x (ndarray): Input features.
            y (ndarray): True labels.
        
        Returns:
            float: Accuracy of the model.
        '''
        output = x
        for layer in range(len(self.layers) - 1):
            output = self.layers[layer].forward(output)
        
        out = self.layers[-1].forward(output)
        y_pred = DataParser.softmax(out)
        
        predicted_labels = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y, axis=1)

        accuracy = np.mean(predicted_labels == true_labels)
        return accuracy


    def evaluate_f1(self, x, y):
        '''
        Evaluates the F1 score of the model on the given data.

        Args:
            x (ndarray): Input features.
            y (ndarray): True labels.
        
        Returns:
            float: F1 score of the model.
        '''
        output = x
        for layer in range(len(self.layers) - 1):
            output = self.layers[layer].forward(output)
        
        out = self.layers[-1].forward(output)
        y_pred = DataParser.softmax(out)
        
        predicted_labels = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y, axis=1)

        true_positive = np.sum((true_labels == 1) & (predicted_labels == 1))
        false_positive = np.sum((true_labels == 0) & (predicted_labels == 1))
        false_negative = np.sum((true_labels == 1) & (predicted_labels == 0))
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


    def save_model(self):
        '''
        Saves the model's architecture, weights, biases, mean, std, and categories to a .npy file.
        '''
        model_data = {
            "layer_sizes": self.layer,
            "weights": [layer.weights for layer in self.layers],
            "biases": [layer.bias for layer in self.layers],
            "mean": self.mean,
            "std": self.std,
            "categories": self.categories
        }
        np.save("saved_model.npy", model_data)


def main():
    '''
    Entry point for the program: parses command-line arguments, initializes and trains the model,
    saves the trained model, and plots training metrics.
    '''
    parser = argparse.ArgumentParser(description='Predicts whether a cancer is malignant or benign')
    parser.add_argument('-l', '--layer', nargs='+', type=int, default=[16, 16] ,help='Layers of model')
    parser.add_argument('-e', '--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('-s', '--loss', default='categoricalCrossEntropy', help='Loss function')
    parser.add_argument('-b', '--batch_size', type=int, default=256, help='Size of the batch')
    parser.add_argument('-r', '--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--adam', action='store_true', help='Use Adam optimizer instead of SGD')
    args = parser.parse_args()

    nn = Network("train.csv", "validation.csv", vars(args))
    nn.standardize()
    nn.create_layers()
    nn.train()
    nn.save_model()
    nn.graphs()

if __name__ == "__main__":
    main()
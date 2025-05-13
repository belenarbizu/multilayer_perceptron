import numpy as np
from layer import Layer
import argparse
from toolkit import DataParser

class Predict:
    '''
    Loads a trained neural network model and performs predictions on new data.

    Attributes:
        data (DataFrame): Raw input data.
        num_data (DataFrame): Numeric features from the input data.
        layer (list): List of hidden layer sizes from the saved model.
        weight (list): List of weight matrices from the saved model.
        bias (list): List of bias vectors from the saved model.
        mean (dict): Mean values used for standardization.
        std (dict): Standard deviation values used for standardization.
        categories (list): List of prediction categories.
        layers (list): List of Layer objects used to rebuild the network.
    '''
    def __init__(self, data_file):
        '''
        Initializes the prediction process by loading input data and the trained model.

        Args:
            data_file (str): Path to the input CSV file for prediction.
        '''
        self.data = DataParser.replace_nan_values(DataParser.open_file(data_file, None))
        self.num_data = self.data.iloc[:,2:]

        try:
            model_data = np.load("saved_model.npy", allow_pickle=True).item()
        except Exception as e:
            print(e)
            exit(1)
        self.layer = model_data["layer_sizes"]
        self.weight = model_data["weights"]
        self.bias = model_data["biases"]
        self.mean = model_data["mean"]
        self.std = model_data["std"]
        self.categories = model_data["categories"]
        self.layers = []


    def standardize(self):
        '''
        Applies the same standardization used during training to the input data.
        '''
        for col in self.num_data.columns:
            if col in self.mean:
                self.num_data[col] = (self.num_data[col] - self.mean[col]) / self.std[col]

    def create_layers(self):
        '''
        Reconstructs the neural network architecture using saved layer sizes and initializes each Layer.
        '''
        self.n_layers = len(self.layer)
        n_inputs = self.num_data.shape[1]

        input_layer = Layer(n_inputs, self.layer[0])
        self.layers.append(input_layer)

        for num in range(1, self.n_layers):
            layer = Layer(self.layer[num - 1], self.layer[num], activation='relu')
            self.layers.append(layer)
        
        output_layer = Layer(self.layer[self.n_layers - 1], 2, activation='softmax')
        self.layers.append(output_layer)


    def predict(self):
        '''
        Performs a forward pass through the network using the loaded model weights and biases.
        '''
        for i in range(len(self.layers)):
            self.layers[i].weights = self.weight[i]
            self.layers[i].bias = self.bias[i]

        x = self.num_data.values
        for layer in self.layers:
            x = layer.forward(x)
        
        prediction = DataParser.softmax(x)
        y_pred = prediction[:,1]
        prediction = np.argmax(prediction, axis=1)
        labels = [self.categories[i] for i in prediction]
        b_label = (self.data.iloc[:,1] == "B").astype(int)
        self.binary_cross_entropy(np.array(b_label, dtype=np.int32), np.array(y_pred, dtype=np.float64))
    
    def binary_cross_entropy(self, y, y_pred):
        '''
        Calculates the binary cross-entropy loss between true labels and predicted probabilities.

        Args:
            y (ndarray): Ground truth binary labels (0 or 1).
            y_pred (ndarray): Predicted probabilities for the positive class.
        '''
        m = len(self.num_data)
        epsilon = 1e-15
        loss = - np.sum(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred)) / m
        print(f"loss: {loss}")


def main():
    '''
    Parses command-line arguments, loads input data, and makes predictions using the saved model.
    '''
    parser = argparse.ArgumentParser(description="Predict using the trained model")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file for prediction")
    args = parser.parse_args()

    predictor = Predict(args.input_file)
    predictor.standardize()
    predictor.create_layers()
    predictor.predict()

if __name__ == "__main__":
    main()
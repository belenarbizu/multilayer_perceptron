import numpy as np
from layer import Layer
import argparse
from toolkit import DataParser

class Predict:
    def __init__(self, data_file):
        self.data = DataParser.replace_nan_values(DataParser.open_file(data_file, 0, 0))

        model_data = np.load("model_data.npy", allow_pickle=True).item()
        self.layer = model_data["layer"]
        self.weight = model_data["weight"]
        self.bias = model_data["bias"]
        self.mean = model_data["mean"]
        self.std = model_data["std"]
        self.categories = model_data["categories"]


    def standardize(self):
        for col in self.data.columns:
            if col in self.mean:
                self.data[col] = (self.data[col] - self.mean[col]) / self.std[col]

    def create_layers(self):
        self.layers = []
        self.n_layers = len(self.layer)
        #n_inputs = self.data.iloc[:,:-2].shape[1]
        n_inputs = 30
        input_layer = Layer(n_inputs, self.layer[0])
        self.layers.append(input_layer)

        for num in range(self.n_layers - 1):
            layer = Layer(self.layer[num], self.layer[num + 1], activation='relu')
            self.layers.append(layer)
        
        output_layer = Layer(self.layer[self.n_layers - 1], 2, activation='softmax')
        self.layers.append(output_layer)


    def predict(self):
        for i in range(len(self.layers)):
            self.layers[i].weights = self.weight[i]
            self.layers[i].bias = self.bias[i]

        x = self.data.values
        for layer in self.layers:
            x = layer.forward(x)
        
        prediction = DataParser.softmax(x)
        prediction = np.argmax(prediction, axis=1)
        labels = [self.categories[i] for i in prediction]
        return labels


def main():
    parser = argparse.ArgumentParser(description="Predict using the trained model")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file for prediction")
    args = parser.parse_args()

    predictor = Predict(args.input_file)
    predictor.standardize()
    predictor.create_layers()
    predictions = predictor.predict()

    print(predictions)

if __name__ == "__main__":
    main()
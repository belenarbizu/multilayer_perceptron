import pandas as pd
import numpy as np


class DataParser:
    @staticmethod
    def open_file(file_path, index_col=None, header=None):
        '''
        Opens a file and returns it.

        Parameters:
        file_path (str): A string path to the file.
        index_col (str, optional): Index column (default is Index).
        Returns:
        data (pd.Series): Dataset of the file.
        '''
        try:
            data = pd.read_csv(file_path, index_col=index_col, header=header)
            return data
        except Exception as e:
            print(e)
            exit(1)


    @staticmethod
    def replace_nan_values(data):
        '''
        Replaces NaN values of a Dataset.

        Parameters:
        data (pd.Series): A dataset with NaN values.
        Returns:
        data (pd.Series): A dataset without NaN values.
        '''
        for col in data.columns:
            if data[col].isnull().any():
                non_nan_values = data[col][data[col].notnull()]
                if not(non_nan_values.empty):
                    column_mean = sum(non_nan_values) / len(non_nan_values)
                    data[col] = data[col].fillna(column_mean)
        return data

    @staticmethod
    def clean_data(data):
        '''
        Removes non-numeric data and replace NaN values.

        Parameters:
        data (pd.Series): A dataset.
        Returns:
        num_data (pd.Series): A dataset without non-numeric data and NaN values.
        nan_data (pd.Series): A dataset without non-numeric data.
        '''
        try:
            columns_name = []
            for name in data.columns:
                if isinstance(data[name].iloc[0], int) or isinstance(data[name].iloc[0], float):
                    columns_name.append(name)
            num_data = data.loc[:, columns_name]
            nan_data = num_data.copy()
            num_data = DataParser.replace_nan_values(num_data)
        except Exception as e:
            print(e)
            exit(1)
        return num_data, nan_data
    
    @staticmethod
    def sigmoid(x):
        '''
        Applies the sigmoid activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Output after applying sigmoid function.
        '''
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def relu(x):
        '''
        Applies the ReLU activation function.

        Parameters:
        x (np.ndarray): Input array.

        Returns:
        np.ndarray: Output after applying ReLU function.
        '''
        return np.maximum(0, x)
    
    @staticmethod
    def softmax(inputs):
        '''
        Applies the softmax activation function.

        Parameters:
        inputs (np.ndarray): Input array.

        Returns:
        np.ndarray: Output after applying softmax function.
        '''
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return output

    @staticmethod
    def relu_derivative(output):
        '''
        Computes the derivative of the ReLU function.

        Parameters:
        output (np.ndarray): Output after applying ReLU.

        Returns:
        np.ndarray: Derivative of ReLU (1 where output > 0, else 0).
        '''
        return (output > 0).astype(float)
        
# Multilayer Perceptron

This project is a from-scratch implementation of a **Multilayer Perceptron (MLP)** used to classify breast cancer cases as **malignant (M)** or **benign (B)** using the **Wisconsin Breast Cancer Diagnostic dataset**. The goal is to deeply understand how neural networks work internally — including forward propagation, backpropagation, and optimization — by building everything manually without using machine learning libraries.

---

## Project Structure

- `train.py`: Main training script with configurable architecture and training options.
- `predict.py`: Loads a saved model and performs predictions on new data.
- `split_data.py`: Splits the dataset into training and validation sets.
- `compare_models.py`: Trains multiple models with different configurations and compares their learning curves.
- `layer.py`: Defines a dense layer with support for ReLU and softmax.
- `toolkit.py`: Utility functions (data loading, normalization, activation functions).
- `train.csv` / `validation.csv`: Processed dataset files.

---

## Features

- Customizable neural network architecture
- Training with backpropagation and softmax + cross-entropy
- Early stopping
- L2 regularization
- Adam optimizer
- F1 Score evaluation
- Visualization of training/validation metrics (loss and accuracy)
- Multiple model comparison with visualization

---

## Training a Model

The main training logic is handled by the `train.py` script. You can configure layer sizes, number of epochs, learning rate, batch size, and optimizer via command-line arguments.

This will:
- Standardize the dataset
- Create the specified network
- Train the model with or without Adam
- Apply L2 regularization and early stopping
- Save the model and training plots

---

## Prediction

A separate prediction script (`predict.py`) loads the saved model and evaluates it on new data, providing the loss.

---

## Model Comparison

The `compare_models.py` script trains multiple models using different configurations and plots their training and validation curves (loss and accuracy) in the same graph for visual comparison.

---

## Getting Started

1. Clone this repository:
```bash
git clone <repo-url>
cd repo-folder
```

2. Prepare the data:
```bash
python3 split_data.py
```

3. Train the model:
```bash
python3 train.py [OPTIONS]
```

**Available options:**

- `-l`, `--layer`: List of integers defining the number of neurons in each hidden layer.  
  _Example_: `--layer 16 8`
- `-e`, `--epochs`: Number of training epochs.  
  _Default_: 150
- `-s`, `--loss`: Loss function (default is `'categoricalCrossEntropy'`).
- `-b`, `--batch_size`: Batch size for training.  
  _Default_: 256
- `-r`, `--learning_rate`: Learning rate for the optimizer.  
  _Default_: 0.001
- `--adam`: Optional flag to enable the Adam optimizer (use SGD by default).

4. Compare different models:
```bash
python3 compare_models.py
```

5. Predict on new data:
```bash
python3 predict.py validation.csv
```

---

## Requirements

- Python 3.x
- NumPy
- matplotlib
- pandas

---

## Technical Details

This project manually implements a feedforward neural network. Below is a summary of the core mathematical operations.


### Forward Propagation

Each layer computes:

- `Z = X · W + b`  
- Activation:
  - ReLU: `A = max(0, Z)`
  - Softmax: `A = exp(Z) / sum(exp(Z))`

### Loss Function — Categorical Cross-Entropy

Used for binary classification with one-hot encoded labels:

- `Loss = -sum(y * log(y_pred))`

Where:
- `y` is the true label (one-hot)
- `y_pred` is the predicted probability from softmax

### Backpropagation

Used to compute gradients and propagate error backward.

- Gradient w.r.t. weights: `dW = Xᵀ · δ`
- Gradient w.r.t. bias: `db = sum(δ)`
- Gradient w.r.t. input: `dX = δ · Wᵀ`
- Delta: `δ = dL/dA * activation_derivative(Z)`

For softmax + cross-entropy:  
- `δ = y_pred - y`

### Gradient Descent (SGD)

Update rules:

- `W := W - η * (dW + λW)`
- `b := b - η * db`

Where:
- `η` is the learning rate
- `λ` is the L2 regularization coefficient

### Adam Optimizer

With moment estimates:

- `m_t = β1 * m_{t-1} + (1 - β1) * grad`
- `v_t = β2 * v_{t-1} + (1 - β2) * grad²`
- Bias correction:
  - `m̂ = m_t / (1 - β1^t)`
  - `v̂ = v_t / (1 - β2^t)`
- Update:
  - `W := W - η * m̂ / (sqrt(v̂) + ε)`

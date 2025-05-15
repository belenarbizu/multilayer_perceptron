from train import Network
import numpy as np
import matplotlib.pyplot as plt

args = [
    {"layer": [16, 4], "epochs": 150, "loss": "categoricalCrossEntropy", "batch_size": 256, "learning_rate": 0.01, "adam": False},
    {"layer": [24, 8], "epochs": 150, "loss": "categoricalCrossEntropy", "batch_size": 256, "learning_rate": 0.001, "adam": True},
    {"layer": [32, 8], "epochs": 300, "loss": "categoricalCrossEntropy", "batch_size": 128, "learning_rate": 0.001, "adam": False},
]

results = []

for i, arg in enumerate(args):
    print(f"\nTraining model: {arg}")
    nn = Network("train.csv", "validation.csv", arg)
    nn.standardize()
    nn.create_layers()
    nn.train()
    
    results.append({
        "name": f"Model {i+1}",
        "train_losses": nn.train_losses,
        "val_losses": nn.val_losses,
        "train_accuracies": nn.train_accuracies,
        "val_accuracies": nn.val_accuracies
    })

fig, axs = plt.subplots(1, 2, figsize=(25, 13))
colors = ['blue', 'orange', 'green']

for i, result in enumerate(results):
    axs[0].plot(result["val_losses"], color=colors[i], label=f"{result['name']} val_loss")
    axs[0].plot(result["train_losses"], linestyle='--', color=colors[i], alpha=0.6, label=f"{result['name']} train_loss")

for i, result in enumerate(results):
    axs[1].plot(result["val_accuracies"], color=colors[i], label=f"{result['name']} val_acc")
    axs[1].plot(result["train_accuracies"], linestyle='--', color=colors[i], alpha=0.6, label=f"{result['name']} train_acc")

axs[0].set_title("Loss")
axs[0].set_xlabel("epochs")
axs[0].set_ylabel("loss")
axs[0].legend()
axs[0].grid(True)

axs[1].set_title("Accuracy")
axs[1].set_xlabel("epochs")
axs[1].set_ylabel("accuracy")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig("comparing_models.png")
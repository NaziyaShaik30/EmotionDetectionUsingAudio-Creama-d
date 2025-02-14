import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from preprocess import load_data
from classifficationModel import DeepANN, SimpleANN

def evaluate_model(model, test_loader, device):
    """Function to evaluate model accuracy on test data."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():  # No gradient calculation
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)  # Get class with highest probability
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    return 100 * correct / total  # Accuracy in percentage

def main():
    # Load dataset
    train_path = r"C:\Users\NAZIYA\PYTHON\DL-SKILL\project\audio_data\crema-d\train"
    test_path = r"C:\Users\NAZIYA\PYTHON\DL-SKILL\project\audio_data\crema-d\test"

    # Check if files exist in the dataset folders
    print("Train files:", len(os.listdir(train_path)))
    print("Test files:", len(os.listdir(test_path)))

    # Load Data
    train_X, train_y = load_data(train_path)
    test_X, test_y = load_data(test_path)

    # Check if data is loaded correctly
    print("Train_X shape:", train_X.shape)  # Expected: (N, 50, 40)
    print("Train_y shape:", train_y.shape)  # Expected: (N,)
    print("Test_X shape:", test_X.shape)    # Expected: (M, 50, 40)
    print("Test_y shape:", test_y.shape)    # Expected: (M,)

    # Convert to PyTorch tensors
    train_X = torch.tensor(train_X, dtype=torch.float32)
    test_X = torch.tensor(test_X, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.long)
    test_y = torch.tensor(test_y, dtype=torch.long)

    # Data loaders
    batch_size = 32
    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=batch_size, shuffle=False)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Debugging: Check batch shape before training
    for X_batch, y_batch in train_loader:
        print("\n🔹 Checking batch shapes before training:")
        print("Batch X shape:", X_batch.shape)  # Expected: (batch_size, 50, 40)
        print("Batch y shape:", y_batch.shape)  # Expected: (batch_size,)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Create a sample model to verify forward pass
        input_size = X_batch.shape[2]  # Should be 40
        num_classes = len(torch.unique(train_y))
        model = DeepANN(input_size, num_classes).to(device)
        output = model(X_batch)  # Forward pass test
        print("Model output shape:", output.shape)  # Should be (batch_size, num_classes)
        break  # Stop after one batch to avoid unnecessary computation

    # Optimizers
    optimizers = {
        "Adam": optim.Adam,
        "RMSprop": optim.RMSprop,
        "SGD": optim.SGD
    }

    # Model selection
    models_dict = {
        "Deep ANN": lambda: DeepANN(input_size, num_classes).to(device),
        "Simple ANN": lambda: SimpleANN(input_size, num_classes).to(device)
    }

    loss_dict = {}
    accuracy_dict = {}

    # Train models
    for model_name, model_fn in models_dict.items():
        for opt_name, opt_class in optimizers.items():
            print(f"\nTraining {model_name} with {opt_name} optimizer...")
            model = model_fn()
            optimizer = opt_class(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            losses = []
            for epoch in range(10):
                epoch_loss = 0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    output = model(X_batch)  # Forward pass
                    loss = criterion(output, y_batch)  # Compute loss
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                losses.append(epoch_loss / len(train_loader))
                print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.4f}")

            loss_dict[f"{model_name}_{opt_name}"] = losses

            accuracy = evaluate_model(model, test_loader, device)
            accuracy_dict[f"{model_name}_{opt_name}"] = accuracy
            print(f"✅ Accuracy of {model_name} with {opt_name}: {accuracy:.2f}%")

    # Plot Loss Graph
    plt.figure()
    for key, loss in loss_dict.items():
        plt.plot(loss, label=key)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss vs. Epochs")
    plt.show()

    print("\n🔹 Final Model Accuracies:")
    for key, acc in accuracy_dict.items():
        print(f"{key}: {acc:.2f}%")


if __name__ == "__main__":
    main()

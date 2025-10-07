#!/usr/bin/env python3
"""
Custom example demonstrating advanced usage of ToSTLinearEngine.

This example shows:
1. Multi-layer linear models with different activation functions
2. Custom training loops with learning rate scheduling
3. Model comparison and evaluation
4. Saving and loading trained models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Import our custom modules
from tostlinear.model import ToSTLinear
from tostlinear.utils import create_sample_data, calculate_loss


class MultiLayerToSTLinear(nn.Module):
    """Multi-layer linear network using ToSTLinear components."""

    def __init__(self, input_size, hidden_sizes, output_size, activations=None):
        super().__init__()
        self.layers = nn.ModuleList()

        # Input layer
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            activation = activations[i] if activations and i < len(activations) else 'relu'
            self.layers.append(ToSTLinear(prev_size, hidden_size, activation_type=activation))
            prev_size = hidden_size

        # Output layer (no activation for regression)
        self.layers.append(ToSTLinear(prev_size, output_size, use_activation=False))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def generate_regression_data(num_samples=1000, input_dim=10, noise=0.1):
    """Generate synthetic regression data."""
    torch.manual_seed(42)
    X = torch.randn(num_samples, input_dim)

    # Create some non-linear relationships
    weights = torch.randn(input_dim, 1)
    y = X @ weights + noise * torch.randn(num_samples, 1)

    return X, y


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test,
                           epochs=200, lr=0.001, save_path=None):
    """Train model and return training history."""

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.8)
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Training
        y_pred = model(X_train)
        train_loss = criterion(y_pred, y_train)
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test)
            test_loss = criterion(y_test_pred, y_test)

        model.train()
        scheduler.step()

        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss.item():.6f} | "
                  f"Test Loss: {test_loss.item():.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save model if path provided
    if save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'training_history': {'train_losses': train_losses, 'test_losses': test_losses}
        }, save_path)

    return train_losses, test_losses


def compare_activations():
    """Compare different activation functions."""

    # Generate data
    X, y = generate_regression_data(1000, 5)
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    # Model configurations
    hidden_sizes = [10, 8]
    activations_list = [
        ['relu', 'relu'],
        ['tanh', 'tanh'],
        ['sigmoid', 'sigmoid'],
        ['leaky_relu', 'leaky_relu']
    ]

    results = {}

    for activations in activations_list:
        print(f"\n--- Training with {activations} activations ---")

        model = MultiLayerToSTLinear(
            input_size=5,
            hidden_sizes=hidden_sizes,
            output_size=1,
            activations=activations
        )

        train_losses, test_losses = train_and_evaluate_model(
            model, X_train, y_train, X_test, y_test,
            epochs=100, lr=0.01
        )

        results[str(activations)] = {
            'final_train_loss': train_losses[-1],
            'final_test_loss': test_losses[-1],
            'min_test_loss': min(test_losses),
            'train_losses': train_losses,
            'test_losses': test_losses
        }

    # Save results
    with open('activation_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n--- Results Summary ---")
    for activation, result in results.items():
        print(f"{activation}: Final Test Loss = {result['final_test_loss']:.6f}")

    return results


def demonstrate_model_persistence():
    """Demonstrate saving and loading trained models."""

    # Generate data
    X, y = generate_regression_data(500, 3)
    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]

    # Train model
    model = MultiLayerToSTLinear(
        input_size=3,
        hidden_sizes=[6, 4],
        output_size=1,
        activations=['relu', 'relu']
    )

    print("\n--- Training model for persistence demo ---")
    train_losses, test_losses = train_and_evaluate_model(
        model, X_train, y_train, X_test, y_test,
        epochs=80, lr=0.01,
        save_path='demo_model.pth'
    )

    # Load and evaluate
    print("\n--- Loading and evaluating saved model ---")
    checkpoint = torch.load('demo_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        final_loss = nn.MSELoss()(y_pred, y_test)
        print(f"Loaded model test loss: {final_loss:.6f}")

    return model


if __name__ == "__main__":
    print("ToSTLinearEngine Custom Examples")
    print("=" * 40)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Run activation comparison
    print("\n1. Comparing Activation Functions")
    print("-" * 30)
    results = compare_activations()

    # Demonstrate model persistence
    print("\n\n2. Model Persistence Demo")
    print("-" * 30)
    model = demonstrate_model_persistence()

    print("\n" + "=" * 40)
    print("Custom examples completed successfully!")

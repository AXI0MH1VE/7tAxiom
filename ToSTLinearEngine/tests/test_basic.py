#!/usr/bin/env python3
"""
Comprehensive test suite for ToSTLinearEngine.

Tests cover:
1. ToSTLinear model functionality
2. Different activation functions
3. Training utilities
4. Edge cases and error handling
5. Model persistence
"""

import unittest
import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path

# Import our modules
from tostlinear.model import ToSTLinear
from tostlinear.utils import create_sample_data, calculate_loss, train_model


class TestToSTLinear(unittest.TestCase):
    """Test cases for ToSTLinear model."""

    def setUp(self):
        """Set up test fixtures."""
        self.input_size = 10
        self.output_size = 5
        self.batch_size = 32

    def test_model_initialization(self):
        """Test model initialization with different configurations."""
        # Test default configuration
        model = ToSTLinear(self.input_size, self.output_size)
        self.assertTrue(model.use_activation)
        self.assertEqual(model.activation_type, 'relu')

        # Test without activation
        model_no_act = ToSTLinear(self.input_size, self.output_size, use_activation=False)
        self.assertFalse(model_no_act.use_activation)

        # Test different activation types
        activations = ['relu', 'sigmoid', 'tanh', 'leaky_relu']
        for activation in activations:
            model = ToSTLinear(self.input_size, self.output_size, activation_type=activation)
            self.assertEqual(model.activation_type, activation)

    def test_invalid_activation_type(self):
        """Test that invalid activation type raises ValueError."""
        with self.assertRaises(ValueError):
            ToSTLinear(self.input_size, self.output_size, activation_type='invalid')

    def test_forward_pass(self):
        """Test forward pass through the model."""
        model = ToSTLinear(self.input_size, self.output_size, use_activation=False)

        # Create test input
        x = torch.randn(self.batch_size, self.input_size)

        # Forward pass
        output = model(x)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_size))

        # Check that no activation was applied (should be different from input)
        self.assertFalse(torch.equal(output, x[:, :self.output_size]))

    def test_forward_pass_with_activation(self):
        """Test forward pass with activation functions."""
        activations = ['relu', 'sigmoid', 'tanh', 'leaky_relu']

        for activation in activations:
            model = ToSTLinear(self.input_size, self.output_size, activation_type=activation)
            x = torch.randn(self.batch_size, self.input_size)

            # Test that model runs without error
            output = model(x)
            self.assertEqual(output.shape, (self.batch_size, self.output_size))

            # For ReLU, check that negative values are zeroed
            if activation == 'relu':
                x_negative = torch.randn(self.batch_size, self.input_size) - 2  # Mostly negative
                output_relu = model.activation(x_negative[:, :self.output_size])
                self.assertTrue(torch.all(output_relu >= 0))

    def test_model_parameters(self):
        """Test that model parameters are properly initialized."""
        model = ToSTLinear(self.input_size, self.output_size)

        # Check that linear layer has parameters
        self.assertTrue(hasattr(model.linear, 'weight'))
        self.assertTrue(hasattr(model.linear, 'bias'))

        # Check parameter shapes
        self.assertEqual(model.linear.weight.shape, (self.output_size, self.input_size))
        self.assertEqual(model.linear.bias.shape, (self.output_size,))


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""

    def test_create_sample_data(self):
        """Test sample data creation."""
        input_size = 10
        batch_size = 5

        data = create_sample_data(input_size, batch_size)

        # Check shape
        self.assertEqual(data.shape, (batch_size, input_size))

        # Check that data is not all zeros
        self.assertFalse(torch.allclose(data, torch.zeros_like(data)))

    def test_calculate_loss(self):
        """Test loss calculation."""
        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)

        loss = calculate_loss(predictions, targets)

        # Check that loss is a scalar tensor
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)  # Loss should be positive

    def test_calculate_loss_with_custom_loss_fn(self):
        """Test loss calculation with custom loss function."""
        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)

        custom_loss_fn = nn.L1Loss()
        loss = calculate_loss(predictions, targets, custom_loss_fn)

        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)

    def test_train_model(self):
        """Test model training function."""
        # Create simple model and data
        model = ToSTLinear(5, 3, use_activation=False)
        data = torch.randn(20, 5)
        targets = torch.randn(20, 3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Train for a few epochs
        initial_params = [p.clone() for p in model.parameters()]
        train_model(model, data, targets, optimizer, epochs=5)

        # Check that parameters changed
        for initial, current in zip(initial_params, model.parameters()):
            self.assertFalse(torch.allclose(initial, current))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""

    def test_end_to_end_training(self):
        """Test complete training workflow."""
        # Setup
        input_size = 8
        output_size = 4
        model = ToSTLinear(input_size, output_size, activation_type='relu')

        # Generate data
        X = torch.randn(100, input_size)
        y = torch.randn(100, output_size)

        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        train_model(model, X, y, optimizer, epochs=10)

        # Test inference
        model.eval()
        with torch.no_grad():
            test_X = torch.randn(10, input_size)
            predictions = model(test_X)
            self.assertEqual(predictions.shape, (10, output_size))

    def test_model_serialization(self):
        """Test saving and loading model state."""
        model = ToSTLinear(10, 5, activation_type='tanh')

        # Train briefly
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        data = torch.randn(20, 10)
        targets = torch.randn(20, 5)
        train_model(model, data, targets, optimizer, epochs=5)

        # Save and load
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name

        try:
            torch.save(model.state_dict(), temp_path)

            # Create new model and load state
            new_model = ToSTLinear(10, 5, activation_type='tanh')
            new_model.load_state_dict(torch.load(temp_path))

            # Check that parameters match
            for old_param, new_param in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.allclose(old_param, new_param))

        finally:
            os.unlink(temp_path)

    def test_multiple_activations_comparison(self):
        """Test that different activations produce different results."""
        input_size, output_size = 5, 3
        x = torch.randn(10, input_size)

        activations = ['relu', 'sigmoid', 'tanh', 'leaky_relu']
        outputs = {}

        for activation in activations:
            model = ToSTLinear(input_size, output_size, activation_type=activation)
            with torch.no_grad():
                outputs[activation] = model(x)

        # Check that different activations produce different outputs
        for i, act1 in enumerate(activations):
            for act2 in activations[i+1:]:
                self.assertFalse(torch.allclose(outputs[act1], outputs[act2], atol=1e-6))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_single_sample(self):
        """Test with single sample input."""
        model = ToSTLinear(5, 3, use_activation=False)
        x = torch.randn(1, 5)  # Single sample

        output = model(x)
        self.assertEqual(output.shape, (1, 3))

    def test_large_model(self):
        """Test with large input/output dimensions."""
        model = ToSTLinear(1000, 500, use_activation=False)
        x = torch.randn(10, 1000)

        output = model(x)
        self.assertEqual(output.shape, (10, 500))

    def test_zero_input(self):
        """Test with zero input."""
        model = ToSTLinear(3, 2, use_activation=False)
        x = torch.zeros(5, 3)

        output = model(x)
        self.assertEqual(output.shape, (5, 2))
        # Output should not be all zeros (unless weights are zero)
        self.assertFalse(torch.allclose(output, torch.zeros_like(output)))


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2)

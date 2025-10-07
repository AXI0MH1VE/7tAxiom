import torch
import torch.nn as nn
import torch.nn.functional as F

class ToSTLinear(nn.Module):
    def __init__(self, input_size, output_size, use_activation=True, activation_type='relu'):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.use_activation = use_activation
        self.activation_type = activation_type
        if use_activation:
            if activation_type == 'relu':
                self.activation = nn.ReLU()
            elif activation_type == 'sigmoid':
                self.activation = nn.Sigmoid()
            elif activation_type == 'tanh':
                self.activation = nn.Tanh()
            elif activation_type == 'leaky_relu':
                self.activation = nn.LeakyReLU()
            else:
                raise ValueError("Invalid activation type. Choose from 'relu', 'sigmoid', 'tanh', 'leaky_relu'")

    def forward(self, x):
        x = self.linear(x)
        if self.use_activation:
            x = self.activation(x)
        return x

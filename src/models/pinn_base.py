import torch  # PyTorch base package for tensor computations
import torch.nn as nn  # Neural network components from PyTorch

# Define the base class for the Physics-Informed Neural Network (PINN)
class PINN(nn.Module):
    """
    Core feedforward network used across all PINN applications.
    It approximates the solution u(t, x) or a vector [u, v] depending on the PDE.
    """
    def __init__(self, layers):
        # Call the parent class constructor
        super().__init__()

        # Define activation function: tanh is preferred in PINNs due to smooth gradients
        self.activation = nn.Tanh()

        # Initialize a list to store fully connected layers
        self.net = nn.ModuleList()

        # Create the layers of the network
        for i in range(len(layers) - 1):
            # Each Linear layer maps from layers[i] → layers[i+1]
            self.net.append(nn.Linear(layers[i], layers[i+1]))

    def forward(self, x):
        """
        Forward pass through the network.
        Applies tanh activation after each layer except the output layer.
        Args:
            x: input tensor with shape [batch_size, input_dim]
        Returns:
            Output tensor representing prediction of u(t, x)
        """
        for layer in self.net[:-1]:  # Apply activation on all but last layer
            x = self.activation(layer(x))
        return self.net[-1](x)  # Final output layer (no activation)

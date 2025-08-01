import torch
import numpy as np
from src.models.pinn_base import PINN
from src.training.trainer import train
from src.pde.burgers import burgers_residual

# ---------------------------------------------------
# This script runs the full training process for Burgers' equation
# ---------------------------------------------------

# 1. Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Define model architecture: [input_dim, hidden1, hidden2, ..., output_dim]
layers = [2, 50, 50, 50, 1]  # 2 inputs (t, x) → 1 output (u)
model = PINN(layers).to(device)

# 3. Load or generate training data (simplified mock data here)
n_data = 100  # number of known data points
x_data = torch.linspace(-1, 1, n_data).reshape(-1, 1)
t_data = torch.zeros_like(x_data)  # t = 0 for initial condition
u_data = -torch.sin(np.pi * x_data)  # example initial condition: u(x, 0) = -sin(pi x)

# 4. Generate collocation points (where PDE must be satisfied)
n_f = 10000
x_f = torch.rand(n_f, 1) * 2 - 1  # range: [-1, 1]
t_f = torch.rand(n_f, 1) * 1      # range: [0, 1]

# 5. Convert all data to tensors with gradients
t_data = t_data.to(device).requires_grad_()
x_data = x_data.to(device).requires_grad_()
u_data = u_data.to(device)
t_f = t_f.to(device).requires_grad_()
x_f = x_f.to(device).requires_grad_()

# 6. Set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 7. Train model
model = train(model, optimizer,
              x_data, t_data, u_data,
              x_f, t_f, nu=0.01/np.pi,
              residual_fn=burgers_residual,
              epochs=5000)

# 8. Save model if needed
# torch.save(model.state_dict(), "burgers_model.pt")

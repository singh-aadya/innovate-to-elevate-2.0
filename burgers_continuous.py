import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Neural Network definition
# ----------------------------
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, t, x):
        input = torch.cat([t, x], dim=1)
        return self.hidden(input)

# ----------------------------
# Residual (PDE error)
# ----------------------------
def burgers_residual(model, x, t, nu):
    x.requires_grad = True
    t.requires_grad = True
    u = model(t, x)

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]

    f = u_t + u * u_x - nu * u_xx
    return f

# ----------------------------
# Training Setup
# ----------------------------
def train():
    # Viscosity
    nu = 0.01 / np.pi

    # Create model and optimizer
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Collocation points (physics loss)
    N_f = 10000
    x_f = torch.FloatTensor(N_f, 1).uniform_(-1, 1)
    t_f = torch.FloatTensor(N_f, 1).uniform_(0, 1)

    # Initial condition: u(x, 0) = -sin(pi * x)
    N_u = 100
    x_u = torch.linspace(-1, 1, N_u).view(-1, 1)
    t_u = torch.zeros_like(x_u)
    u_u = -torch.sin(np.pi * x_u)

    x_u.requires_grad = True
    t_u.requires_grad = True

    for epoch in range(10000):
        optimizer.zero_grad()

        # Data loss (initial condition)
        u_pred = model(t_u, x_u)
        mse_u = torch.mean((u_pred - u_u) ** 2)

        # Physics loss
        f_pred = burgers_residual(model, x_f, t_f, nu)
        mse_f = torch.mean(f_pred ** 2)

        # Total loss
        loss = mse_u + mse_f
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    return model

# ----------------------------
# Run Training
# ----------------------------
if __name__ == "__main__":
    trained_model = train()

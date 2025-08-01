import torch
import torch.nn as nn

# --------------------------------------------------------------
# Trainer for solving Burgers' equation using a PINN (Physics-Informed NN)
# This loop teaches the model to match both real data and the physics
# --------------------------------------------------------------
def train(model, optimizer, x_data, t_data, u_data, x_colloc, t_colloc, nu, residual_fn, epochs=5000):
    """
    model:        the neural network that learns u(t, x)
    optimizer:    optimizer (e.g., Adam)
    x_data, t_data, u_data: known data points (supervised part)
    x_colloc, t_colloc: random points in space-time (for physics loss)
    nu:           viscosity constant for the PDE
    residual_fn:  function that computes PDE residual (like burgers_residual)
    epochs:       number of training loops to run
    """

    # Mean Squared Error loss function for both data and physics
    mse_loss = nn.MSELoss()

    for epoch in range(epochs):
        # Clear gradients from last step
        optimizer.zero_grad()

        # ----------------------
        # DATA LOSS (MSE between true u and predicted u)
        # ----------------------
        input_data = torch.cat([t_data, x_data], dim=1)  # Combine inputs
        pred_u = model(input_data)  # Predict u(t, x)
        loss_u = mse_loss(pred_u, u_data)  # Compare with true values

        # ----------------------
        # PHYSICS LOSS (residual of PDE should be near zero)
        # ----------------------
        f = residual_fn(model, x_colloc, t_colloc, nu)  # Get PDE error at collocation points
        loss_f = mse_loss(f, torch.zeros_like(f))  # We want f ≈ 0

        # ----------------------
        # TOTAL LOSS = data + physics
        # ----------------------
        loss = loss_u + loss_f

        # Backpropagation: update weights to reduce loss
        loss.backward()
        optimizer.step()

        # Print progress every 500 epochs
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}, Data Loss = {loss_u.item():.6f}, Physics Loss = {loss_f.item():.6f}")

    print("Training finished.")
    return model

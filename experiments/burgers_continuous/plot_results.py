import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.pinn_base import PINN

# ---------------------------------------------------
# Visualize the model predictions for u(t, x)
# Assumes model is already trained
# ---------------------------------------------------

def plot_predictions(model, device):
    model.eval()  # Set model to evaluation mode

    # Create a grid of (t, x) values
    t_vals = np.linspace(0, 1, 100)
    x_vals = np.linspace(-1, 1, 100)
    T, X = np.meshgrid(t_vals, x_vals)
    t_flat = T.flatten().reshape(-1, 1)
    x_flat = X.flatten().reshape(-1, 1)

    # Convert to torch tensors
    t_tensor = torch.tensor(t_flat, dtype=torch.float32, requires_grad=True).to(device)
    x_tensor = torch.tensor(x_flat, dtype=torch.float32, requires_grad=True).to(device)
    input_tensor = torch.cat([t_tensor, x_tensor], dim=1)

    # Predict u(t, x) using the trained model
    with torch.no_grad():
        u_pred = model(input_tensor).cpu().numpy()

    # Reshape predictions back to grid shape
    U = u_pred.reshape(100, 100)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.contourf(T, X, U, 100, cmap='jet')
    plt.colorbar(label='u(t,x)')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Predicted Solution u(t,x) by PINN')
    plt.tight_layout()
    plt.savefig("pinn_burgers_prediction.png")
    plt.show()


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = PINN([2, 50, 50, 50, 1]).to(device)
#     model.load_state_dict(torch.load("burgers_model.pt"))
#     plot_predictions(model, device)

import torch  # PyTorch library helps in building and training neural networks

# ------------------------------------------------------------
# This function calculates how much the neural network's prediction
# violates the Burgers' differential equation.
# The goal is to minimize this error during training.
# ------------------------------------------------------------
def burgers_residual(model, x, t, nu):
    """
    Calculates the error (residual) of the Burgers' equation:
        u_t + u * u_x - nu * u_xx = 0

    model: the trained neural network (predicts u based on x and t)
    x: position values (input to the model)
    t: time values (input to the model)
    nu: a small constant (viscosity in fluid flow)
    """

    # Combine time and position into a single input, because the model expects both together
    input = torch.cat([t, x], dim=1)  # Example: if t = 0.5 and x = 1.0 → input = [0.5, 1.0]

    # Ask the neural network to predict u(t, x) based on current input
    u = model(input)  # This is the output of the neural network

    # Compute how u changes with time using automatic differentiation
    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]  # This gives ∂u/∂t

    # Compute how u changes with space (position)
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]  # This gives ∂u/∂x

    # Compute how u_x changes with space again to get second derivative
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True
    )[0]  # This gives ∂²u/∂x²

    # Now plug into the Burgers' equation:
    # If u_t + u * u_x - nu * u_xx = 0 → there's no error
    # If it's not 0 → the model's prediction violates the physics
    f = u_t + u * u_x - nu * u_xx  # This is the residual (error)

    # Return the residual to be used in the loss function
    return f

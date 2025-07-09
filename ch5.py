import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from matplotlib import pyplot as plt

# =============================
# Data Preparation
# =============================

# Target values in Celsius
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]

# Input values in Fahrenheit
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

# Convert lists to PyTorch tensors and reshape to column vectors
t_c = torch.tensor(t_c).unsqueeze(1)
t_u = torch.tensor(t_u).unsqueeze(1)

# Normalize inputs for better training stability
t_un = 0.1 * t_u

# Split data into training and validation sets (80-20 split)
n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)
train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

# Prepare the actual train and validation sets
t_un_train = t_un[train_indices]
t_un_val = t_un[val_indices]
t_c_train = t_c[train_indices]
t_c_val = t_c[val_indices]

# =============================
# Training Loop Definition
# =============================

def training_loop(n_epochs, optimizer, model, loss_fn, t_u_train, t_u_val, t_c_train, t_c_val):
    """Generic training loop for any given model and dataset."""
    for epoch in range(1, n_epochs + 1):

        # Forward pass on training data
        t_p_train = model(t_u_train)
        loss_train = loss_fn(t_p_train, t_c_train)

        # Forward pass on validation data
        t_p_val = model(t_u_val)
        loss_val = loss_fn(t_p_val, t_c_val)

        # Zero out previous gradients to avoid accumulation
        optimizer.zero_grad()

        # Backward pass: compute gradients
        loss_train.backward()

        # Update model parameters
        optimizer.step()

        # Periodically print loss metrics
        if epoch == 1 or epoch % 1000 == 0:
            print('Epoch {}, Training loss {}, Validation loss {}'.format(
                epoch, float(loss_train), float(loss_val)))

# =============================
# Linear Regression Baseline
# =============================

linear_model = nn.Linear(1, 1)  # Simple linear model
optimizer = optim.SGD(linear_model.parameters(), lr=1e-2)

training_loop(
    n_epochs=3000,
    optimizer=optimizer,
    model=linear_model,
    loss_fn=nn.MSELoss(),
    t_u_train=t_un_train,
    t_u_val=t_un_val,
    t_c_train=t_c_train,
    t_c_val=t_c_val
)

# Display learned parameters
print("\nLinear model weight:", linear_model.weight)
print("Linear model bias:", linear_model.bias)

# =============================
# Neural Network with nn.Sequential
# =============================

seq_model = nn.Sequential(
    nn.Linear(1, 13),
    nn.Tanh(),
    nn.Linear(13, 1)
)

optimizer = optim.SGD(seq_model.parameters(), lr=1e-3)

training_loop(
    n_epochs=5000,
    optimizer=optimizer,
    model=seq_model,
    loss_fn=nn.MSELoss(),
    t_u_train=t_un_train,
    t_u_val=t_un_val,
    t_c_train=t_c_train,
    t_c_val=t_c_val
)

# Evaluate model predictions and inspect gradients
print('Model predictions on validation:', seq_model(t_un_val))
print('Actual validation targets:', t_c_val)
print('First layer weight gradients:', seq_model[0].weight.grad)

# =============================
# Plot Model Predictions
# =============================

# Generate inputs for smooth plotting curve
t_range = torch.arange(20., 90.).unsqueeze(1)

# Plot original data, model predictions, and fitted curve
fig = plt.figure(dpi=600)
plt.xlabel("Fahrenheit")
plt.ylabel("Celsius")
plt.plot(t_u.numpy(), t_c.numpy(), 'o', label='Original Data')
plt.plot(t_range.numpy(), seq_model(0.1 * t_range).detach().numpy(), 'c-', label='Model Prediction')
plt.plot(t_u.numpy(), seq_model(0.1 * t_u).detach().numpy(), 'kx', label='Predicted Points')
plt.legend()
plt.show()

# =============================
# Named Sequential Model
# =============================

namedseq_model = nn.Sequential(OrderedDict([
    ('hidden_linear', nn.Linear(1, 12)),
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(12, 1))
]))

print(namedseq_model)

# Print model parameter names and shapes
for name, param in namedseq_model.named_parameters():
    print(name, param.shape)

# Direct access to specific parameter
print("Output layer bias:", namedseq_model.output_linear.bias)

# =============================
# Subclassed Model Definition
# =============================

class SubclassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_linear = nn.Linear(1, 13)
        self.hidden_activation = nn.Tanh()
        self.output_linear = nn.Linear(13, 1)

    def forward(self, input):
        hidden_t = self.hidden_linear(input)
        activated_t = self.hidden_activation(hidden_t)
        output_t = self.output_linear(activated_t)
        return output_t

# Instantiate and display the subclassed model
subclass_model = SubclassModel()
print(subclass_model)

# =============================
# Model Parameter Inspection
# =============================

for type_str, model in [('seq', seq_model), ('namedseq', namedseq_model), ('subclass', subclass_model)]:
    print(f"Model type: {type_str}")
    for name_str, param in model.named_parameters():
        print(f"{name_str:21} {str(param.shape):19} {param.numel()} parameters")
    print()

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import scipy.io
from pyDOE import lhs
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

# =======================
# 1. Fix seeds for repeatability
# =======================
np.random.seed(1234)
torch.manual_seed(1234)
print("[INFO] Seeds fixed.")

# =======================
# 2. Load dataset
# =======================
data = scipy.io.loadmat('burgers_shock.mat')
t = data['t'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact = np.real(data['usol']).T
print("[INFO] Dataset loaded:", Exact.shape)

# Meshgrid
X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact.flatten()[:, None]

# Domain bounds
lb = X_star.min(0)
ub = X_star.max(0)

# =======================
# 3. Prepare training data
# =======================
N_u = 100
N_f = 10000

# Boundary and initial condition points exactly like Raissi
xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
uu1 = Exact[0:1, :].T
xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
uu2 = Exact[:, 0:1]
xx3 = np.hstack((X[:, -1:], T[:, -1:]))
uu3 = Exact[:, -1:]

X_u_train = np.vstack([xx1, xx2, xx3])
u_train = np.vstack([uu1, uu2, uu3])

# Collocation points
X_f_train = lb + (ub - lb) * lhs(2, N_f)
X_f_train = np.vstack((X_f_train, X_u_train))

# Randomly select N_u points
idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
X_u_train = X_u_train[idx, :]
u_train = u_train[idx, :]

# Convert to tensors
X_u_train = torch.tensor(X_u_train, dtype=torch.float32, requires_grad=True)
u_train = torch.tensor(u_train, dtype=torch.float32)
X_f_train = torch.tensor(X_f_train, dtype=torch.float32, requires_grad=True)
print("[INFO] Training data prepared.")

# =======================
# 4. Define PINN model
# =======================
class PINN_Burgers(nn.Module):
    def __init__(self, layers):
        super(PINN_Burgers, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = torch.tanh
        self.init_weights()

    def init_weights(self):
        for m in self.layers:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        out = x
        for i in range(len(self.layers) - 1):
            out = self.activation(self.layers[i](out))
        out = self.layers[-1](out)
        return out

    def net_f(self, x):
        t = x[:, 1:2]
        x_ = x[:, 0:1]
        u = self.forward(torch.cat([x_, t], dim=1))
        u_t = autograd.grad(u, t, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = autograd.grad(u, x_, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = autograd.grad(u_x, x_, torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        nu = 0.01 / np.pi
        return u_t + u * u_x - nu * u_xx

# Model setup
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
model = PINN_Burgers(layers)
print("[INFO] Model initialized.")

# =======================
# 5. Define loss and optimizer
# =======================
def closure():
    optimizer.zero_grad()
    u_pred = model(X_u_train)
    mse_u = torch.mean((u_train - u_pred) ** 2)
    f_pred = model.net_f(X_f_train)
    mse_f = torch.mean(f_pred ** 2)
    loss = mse_u + mse_f
    loss.backward()
    return loss

optimizer = torch.optim.LBFGS(model.parameters(),
                               max_iter=50000,
                               tolerance_grad=1e-8,
                               tolerance_change=1e-9,
                               history_size=100,
                               line_search_fn='strong_wolfe')

# =======================
# 6. Train model
# =======================
print("[INFO] Starting training...")
start_time = time.time()
optimizer.step(closure)
elapsed = time.time() - start_time
print(f"[INFO] Training complete in {elapsed:.2f} seconds.")

# =======================
# 7. Predict solution
# =======================
X_star_tensor = torch.tensor(X_star, dtype=torch.float32)
u_pred = model(X_star_tensor).detach().numpy()
U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
print("[INFO] Prediction complete.")

# =======================
# 8. Plot Figure A.6
# =======================
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(2, 1, 1)

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

ax.plot(X_u_train.detach().numpy()[:, 1],
        X_u_train.detach().numpy()[:, 0], 'kx',
        label=f'Data ({u_train.shape[0]} points)', markersize=4, clip_on=False)

line = np.linspace(x.min(), x.max(), 2)[:, None]
ax.plot(t[25] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[75] * np.ones((2, 1)), line, 'w-', linewidth=1)

ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.legend(frameon=False, loc='best')
ax.set_title('$u(t,x)$')

# Bottom panel
gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=0.45, bottom=0.05, left=0.1, right=0.9, wspace=0.5)

times = [25, 50, 75]
labels = [r'$t = 0.25$', r'$t = 0.50$', r'$t = 0.75$']

for i, (idx_t, label) in enumerate(zip(times, labels)):
    ax = plt.subplot(gs1[0, i])
    ax.plot(x, Exact[idx_t, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred[idx_t, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title(label)
    if i == 1:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=2, frameon=False)

plt.show()
print("[INFO] Plotting complete.")

# ---------------------------- #
#   Import required libraries  #
# ---------------------------- #

import tensorflow as tf                 # TensorFlow for building and training the neural network
import numpy as np                      # NumPy for numerical computations
import matplotlib.pyplot as plt         # Matplotlib for visualization
import scipy.io                         # For loading MATLAB .mat data files
from scipy.interpolate import griddata  # For interpolating scattered data onto a grid

from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time                             # For measuring training duration

# ----------------------------- #
#  Set random seeds for repeatability  #
# ----------------------------- #

np.random.seed(1234)           # Set NumPy seed to ensure consistent results
tf.set_random_seed(1234)       # Set TensorFlow seed for reproducibility

# ----------------------------- #
#  Define the PINN class       #
# ----------------------------- #

class PhysicsInformedNN:
    def __init__(self, X_u, u, X_f, layers, lb, ub):
        """
        Constructor to initialize the PINN model.

        Parameters:
        - X_u : Training input data for initial and boundary conditions
        - u   : Target solution values at X_u
        - X_f : Collocation points for enforcing the physics (residual)
        - layers : Network architecture (number of neurons per layer)
        - lb, ub : Lower and upper bounds for input domain
        """
        self.lb = lb
        self.ub = ub
        self.X_u = X_u
        self.u = u
        self.X_f = X_f
        self.layers = layers

        # Normalize inputs to [-1, 1] for better neural network performance
        self.X_u_tf = self.normalize_tf(X_u)
        self.X_f_tf = self.normalize_tf(X_f)
        self.u_tf = tf.convert_to_tensor(u, dtype=tf.float32)

        # Initialize neural network weights and biases
        self.weights, self.biases = self.initialize_NN(layers)

        # Optimizer (Adam for fast convergence)
        self.optimizer = tf.keras.optimizers.Adam()

        # For timing purposes
        self.start_time = time.time()

    def normalize_tf(self, X):
        """Normalize input to [-1, 1] range using domain bounds."""
        return 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

    def initialize_NN(self, layers):
        """Initialize weights and biases for each layer using Xavier initialization."""
        weights = []
        biases = []
        for l in range(len(layers) - 1):
            W = tf.Variable(tf.random.truncated_normal([layers[l], layers[l+1]], stddev=np.sqrt(2 / (layers[l] + layers[l+1]))), dtype=tf.float32)
            b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def neural_net(self, X):
        """Forward pass through the neural network."""
        num_layers = len(self.weights) + 1
        H = X
        for l in range(num_layers - 2):
            W = self.weights[l]
            b = self.biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))  # tanh activation for hidden layers
        W = self.weights[-1]
        b = self.biases[-1]
        return tf.add(tf.matmul(H, W), b)  # No activation in output layer

    def net_u(self, X):
        """Predict u(t, x) from the neural network."""
        return self.neural_net(X)

    def net_f(self, X):
        """
        Compute the PDE residual f = u_t + u * u_x - nu * u_xx.
        This is the core of PINN — using automatic differentiation.
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)
            u = self.net_u(X)
            u_x = tape.gradient(u, X)[:, 1:2]      # ∂u/∂x
            u_t = tape.gradient(u, X)[:, 0:1]      # ∂u/∂t
        u_xx = tape.gradient(u_x, X)[:, 1:2]       # ∂²u/∂x²
        del tape
        return u_t + u * u_x - (0.01 / np.pi) * u_xx  # Burgers’ equation residual

    def loss(self):
        """
        Total loss: combination of data loss and PDE residual loss.
        Encourages both data fitting and physics compliance.
        """
        u_pred = self.net_u(self.X_u_tf)              # Predicted u at data points
        f_pred = self.net_f(self.X_f_tf)              # Predicted residual at collocation points

        # Mean squared error between predictions and actual values
        loss_u = tf.reduce_mean(tf.square(self.u_tf - u_pred))
        loss_f = tf.reduce_mean(tf.square(f_pred))

        return loss_u + loss_f

    @tf.function
    def train_step(self):
        """One step of optimization using automatic differentiation."""
        with tf.GradientTape() as tape:
            loss_value = self.loss()
        gradients = tape.gradient(loss_value, self.weights + self.biases)
        self.optimizer.apply_gradients(zip(gradients, self.weights + self.biases))

    def train(self, nIter):
        """Train the neural network for a given number of iterations."""
        for it in range(nIter):
            self.train_step()

            if it % 100 == 0:  # Log every 100 steps
                elapsed = time.time() - self.start_time
                loss_value = self.loss()
                print(f"Iteration {it:05d}: Loss = {loss_value:.3e}  Time Elapsed = {elapsed:.2f} seconds")
                self.start_time = time.time()

    def predict(self, X_star):
        """Predict the solution u and residual f at given input X_star."""
        X_star_tf = self.normalize_tf(X_star)
        u_star = self.net_u(X_star_tf)
        f_star = self.net_f(X_star_tf)
        return u_star.numpy(), f_star.numpy()

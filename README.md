
# Physics-Informed Neural Networks (PINNs)

Physics-Informed Neural Networks (PINNs) are a **deep learning framework** designed to solve scientific computing problems by combining neural networks with **governing physical laws**, typically expressed as nonlinear partial differential equations (PDEs).

---

## **Core Concept and Purpose**

PINNs are trained to solve supervised learning tasks while **respecting physical laws**.  
They are particularly effective in the **small data regime**, where data acquisition is costly or incomplete.

<img width="848" height="259" alt="image" src="https://github.com/user-attachments/assets/61dbe875-9b7d-43b3-92d1-61a7fda37d32" />




**Why PINNs?**
- Traditional ML lacks robustness and convergence guarantees in physics-based scenarios.
- Incorporating **prior physical knowledge** acts as a **regularization mechanism**, constraining possible solutions and improving generalization.



---
<img width="828" height="658" alt="image" src="https://github.com/user-attachments/assets/8c1544bd-0ddc-4dd9-a8fb-6391343e57d2" />

## **Key Mechanisms and Components**

1. **Universal Function Approximation**  
   - Use deep neural networks to approximate nonlinear functions **without prior assumptions** or discretization.

2. **Automatic Differentiation (Autograd)**  
   - Frameworks like PyTorch/TensorFlow allow direct computation of derivatives (e.g., u_t, u_xx) from network outputs.
   - Enables formulation of the **physics-informed residual**.

3. **Physics-Informed Residual Network**  
   - Let u(t, x) be the neural network approximation.  
   - Define f(t, x) as the **PDE residual** (e.g., f = u_t + N[u]).  
   - Computed via **chain rule** using autograd.

4. **Custom Loss Functions**  
   - **MSE_u**: Enforces agreement with initial/boundary data.  
   - **MSE_f**: Enforces PDE constraints at collocation points.  
   - Loss = MSE_u + MSE_f.

5. **Regularization Effect**  
   - PDE term in loss improves generalization even with **small datasets**.

6. **Activation Functions**  
   - Common: **tanh** (smooth, differentiable, effective for PDEs).

---
<img width="1415" height="545" alt="image" src="https://github.com/user-attachments/assets/c4e54512-ee8c-4910-a02b-4dbdfaf6f78b" />

## **Types of PINN Algorithms**

### 1. Continuous Time Models
- Define residual f(t, x) from PDE's LHS.  
- Require **collocation points** in the spatio-temporal domain.  
- **Optimization**:  
  - Small datasets → **L-BFGS** (full-batch quasi-Newton).  
  - Large datasets → **SGD/Adam** (mini-batch).  
- **Limitation**: Scaling to high dimensions requires exponentially more collocation points.

### 2. Discrete Time Models
- Use **Runge-Kutta** time-stepping (explicit/implicit).  
- Avoid collocation points.  
- Can take **large time steps** with high stability and accuracy.  
- Solve **entire spatio-temporal solution in one step**.  
- Loss: Sum of squared errors (SSE) over time snapshots.

---
<img width="1696" height="406" alt="image" src="https://github.com/user-attachments/assets/23a61cab-0bb3-4bee-a6c4-c0c52040c3e5" />

<img width="768" height="223" alt="image" src="https://github.com/user-attachments/assets/37cb3d48-5cde-4a75-99fe-b502b491da6a" />

<img width="915" height="306" alt="image" src="https://github.com/user-attachments/assets/17721898-0c86-4bf4-868b-351d58af2a58" />

## **Applications**

- **Quantum Mechanics**: Nonlinear Schrödinger equation (periodic BCs, complex-valued solutions).  
- **Reaction-Diffusion Systems**: Allen-Cahn equation (nonlinearities).  
- **Fluid Dynamics**: Navier–Stokes (parameter identification, pressure reconstruction).  
- **Shallow-Water Waves**: Korteweg–de Vries (parameter estimation, sparse temporal data).  
- **Burgers’ Equation**: Benchmark problem (accuracy, noise robustness).

**Advantages Across Applications**:
- High predictive accuracy.
- Robust to noisy/scattered data.
- Outperforms Gaussian processes for PDE solutions.

---
<img width="1050" height="337" alt="image" src="https://github.com/user-attachments/assets/604c8b18-84cf-4dc0-8ec7-d44fe5ca2f11" />

<img width="708" height="542" alt="image" src="https://github.com/user-attachments/assets/ae40b32d-286b-40b7-9dd8-0ad5b5d2f819" />

## **Open Questions & Future Research**

- **Architecture**: Optimal depth/width for different PDE types?  
- **Data Needs**: Minimum data for stable training?  
- **Optimization**: Why unique parameter convergence?  
- **Vanishing Gradients**: Can activation functions be improved?  
- **Initialization & Normalization**: Better schemes for stability?  
- **Loss Functions**: Beyond MSE/SSE?  
- **Uncertainty Quantification**: How to measure prediction confidence?

---

**Note**:  
PINNs are **not** replacements for classical numerical methods (e.g., FEM, spectral methods).  
They are **complementary**, offering synergy between **data-driven learning** and **classical solvers**.

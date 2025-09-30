# Numerical Integration Truncation Error Analysis

A comprehensive Python toolkit for analyzing truncation errors in numerical integration methods, and ordinary differential equation with applications to physics problems like spring force calculations and decay process respectively.

This project includes:
- **Theoretical analysis** of truncation errors for Riemann sums, Trapezoidal rule, and Simpson's rule
- **Empirical error calculations** and convergence rate analysis
- **Comparative studies** across different function types
- **Visualization tools** for error behavior and convergence patterns

### Core Analysis
- **Truncation Error Formulas**: Mathematical derivation of error bounds
- **Empirical Error Calculation**: Actual error computation for numerical methods
- **Convergence Rate Analysis**: Verification of theoretical convergence orders
- **Function Comparison**

### Numerical Methods Implemented
- **Riemann Sum** (Left Endpoint) - O(h) convergence
- **Trapezoidal Rule** - O(h²) convergence  
- **Simpson's Rule** - O(h⁴) convergence

Radioactive decay simulation
-**main_decay.py**     # Main simulation script
-**ode_solvers.py**     # Euler and RK4 implementations
-**physics_problems.py**  # Analytical solutions and ODE definitions

Physics_problems.py
**decay_process(t, N, lam): ODE function dN/dt = -λN**
**decay_analytic(t, N0, lam): Analytical solution N(t) = N₀e^(-λt)**
o**de_solvers.py**

**euler_method(f, t_span, y0, dt): First-order explicit method**
**rk4_method(f, t_span, y0, dt): Fourth-order Runge-Kutta method**

### Prerequisites
```bash
Python3
NumPy
Matplotlib
SciPy
SymPy

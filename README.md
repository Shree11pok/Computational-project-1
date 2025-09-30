# Numerical Integration Truncation Error Analysis

A comprehensive Python toolkit for analyzing truncation errors in numerical integration methods, with applications to physics problems like spring force calculations.

## 📋 Overview

This project provides:
- **Theoretical analysis** of truncation errors for Riemann sums, Trapezoidal rule, and Simpson's rule
- **Empirical error calculations** and convergence rate analysis
- **Comparative studies** across different function types
- **Visualization tools** for error behavior and convergence patterns

## 🚀 Features

### Core Analysis
- **Truncation Error Formulas**: Mathematical derivation of error bounds
- **Empirical Error Calculation**: Actual error computation for numerical methods
- **Convergence Rate Analysis**: Verification of theoretical convergence orders
- **Function Comparison**: Performance across linear, quadratic, cubic, and trigonometric functions

### Numerical Methods Implemented
- **Riemann Sum** (Left Endpoint) - O(h) convergence
- **Trapezoidal Rule** - O(h²) convergence  
- **Simpson's Rule** - O(h⁴) convergence

### Physics Applications
- Spring force and work calculations
- Hooke's Law integration analysis
- Derivative analysis for error prediction

##radioactive-decay-simulation/
├── main_decay.py              # Main simulation script
├── ode_solvers.py             # Euler and RK4 implementations
├── physics_problems.py        # Analytical solutions and ODE definitions
└── README.md                  # This file

#physics_problems.py
decay_process(t, N, lam): ODE function dN/dt = -λN
decay_analytic(t, N0, lam): Analytical solution N(t) = N₀e^(-λt)
ode_solvers.py

euler_method(f, t_span, y0, dt): First-order explicit method
rk4_method(f, t_span, y0, dt): Fourth-order Runge-Kutta method

## 🛠 Installation

### Prerequisites
```bash
Python 3.7+
NumPy
Matplotlib
SciPy
SymPy

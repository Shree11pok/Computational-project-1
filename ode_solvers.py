#Radioactive Decay ODE Module
"""This module solves the radioactive decay differentail equation:
dN/dt = -λN"
  where:
  N is the quantity that decays over time t
  λ is the decay constant
  Analytic solution is N(t) = N0 * exp(-λ t)"""

#solving for for the ODES using Euler's method and RK4 method
#Scipy : for comparison
#Error analysis between numerical and analytic solutions


import numpy as np

#Euler's method 
def euler_method(f, t_span, y0, dt):
    t0, tf = t_span
    num_steps = int((tf - t0) / dt) + 1
    t_values = np.linspace(t0, tf, num_steps)
    y_values = np.zeros(num_steps)
    y_values[0] = y0 # Initial condition
    
    #Euler method loop
    for i in range(num_steps - 1):
        y_values[i + 1] = y_values[i] + dt * f(t_values[i], y_values[i])

    return t_values, y_values

#Runge-Kutta 4th order method
def rk4_method(f, t_span, y0, dt):
    t0, tf = t_span
    num_steps = int((tf - t0) / dt) + 1
    t_values = np.linspace(t0, tf, num_steps)
    y_values = np.zeros(num_steps)
    y_values[0] = y0 # Initial condition

    #RK4 method loop
    for i in range(num_steps - 1):
        t = t_values[i]
        y = y_values[i]
        
        k1 = dt * f(t, y)
        k2 = dt * f(t + dt / 2, y + k1 / 2)
        k3 = dt * f(t + dt / 2, y + k2 / 2)
        k4 = dt * f(t + dt, y + k3)
        
        y_values[i + 1] = y + (k1 + 2*k2 + 2*k3 + k4) / 6

    return t_values, y_values
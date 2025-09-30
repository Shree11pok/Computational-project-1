# integration_methods.py
"""
Numerical integration methods for definite integrals.

This module implements Riemann sum, Trapezoidal rule, and Simpson's rule,
with comparisons to SciPy implementations.
"""

import numpy as np

def riemann_sum(f, a, b, n= 1000):
    """ computing the integral of f from a to b using Riemann sum
    f : functtion to integrate (callable)
    a, b : integration limits (float)
    n : number of subintervals (int)
    returns: integral value (float)
    """
    x = np.linspace(a,b,n+1)[:-1] # left endpoints
    dx = (b - a) / n
    return np.sum(f(x) * dx)

def trapezoidal_rule(f, a, b, n=1000):
    """ computing the integral of f from a to b using Trapezoidal rule
    f : functtion to integrate (callable)
    a, b : integration limits (float)
    n : number of subintervals (int)
    returns: integral value (float)
    """
    x = np.linspace(a, b, n+1)
    y = f(x)
    dx = (b - a) / n
    return (dx / 2) * np.sum(y[:-1] + y[1:])

def simpsons_rule(f, a, b, n=1000):
    """ computing the integral of f from a to b using Simpson's rule
    f : functtion to integrate (callable)
    a, b : integration limits (float)
    n : number of subintervals (int, must be even)
    returns: integral value (float)
    """
    if n % 2 == 1:
        n += 1  # make n even if it's odd
    x = np.linspace(a, b, n+1)
    y = f(x)
    dx = (b-a)/n
    return (dx / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
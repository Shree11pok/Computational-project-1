import numpy as np

def spring_force(k, x):
    """Spring force (Hooke's law)"""
    return -k * x

def work_done_by_spring(k, a, b, n=1000):
    """Analytical solution for work done by spring from x=a to x=b."""
    return 0.5 * k * (a**2 - b**2)


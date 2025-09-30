
import numpy as np
import matplotlib.pyplot as plt

def decay_process(t, N, lam):
	"""Exponential decay function
	t : time float
	N : current value float
	lam :decay_constant 
	returns: dN/dt (derivative at (t, N) float
	"""
	return -lam * N

def decay_analytic(t, N0, lam):
	"""Analytic solution for exponential decay
		N(t) = N0 * exp(-λ t)
		t : time float
		N0 : initial value float
		lam : decay constant float
		returns: N(t) (float)
	"""
	return N0 * np.exp(-lam * t)
	


 


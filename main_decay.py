"""Simulations of decay process with euler, Rk4 and SciPy solvers"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from ode_solvers import euler_method, rk4_method
from physics_problmes import decay_process, decay_analytic

def run_decay(N0, lam, dt, t_span):
    """decay simulations fot the given method
    N0: initial quantity (float)
    lam: decay constant (float)
    T : final time
    dt: time step size (float)
    method: 'euler', 'rk4', or 'scipy' (string)

    returns: 
    t_values: array of time values
    N_values: array of quantity values  
    N_analytic: array of analytic solution values
    """
    # Define the ODE function dN/dt = -λN
    def f(t,N):
        return decay_process(t, N, lam)
    
     # Euler method
    print(f"Running Euler method...")
    t_euler, N_euler = euler_method(f, t_span, N0, dt)
    print(f"Euler method completed: {len(t_euler)} points")

    # RK4 method
    print(f"Running RK4 method...")
    t_rk4, N_rk4 = rk4_method(f, t_span, N0, dt)
    print(f"RK4 method completed: {len(t_rk4)} points")
    
    # SciPy's solve_ivp method
    print(f"Running SciPy solve_ivp method...")
    try:
        t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
        sol = solve_ivp(f, t_span, [N0], method='RK45', t_eval=t_eval)
        t_scipy = sol.t
        N_scipy = sol.y[0]
        print(f"SciPy solver completed: {len(t_scipy)} points")
    except Exception as e:
        print(f"Error using solve_ivp: {e}")
        t_scipy, N_scipy = t_rk4, N_rk4  # Fallback to RK4 results

    # Analytic solution
    t_analytic = np.arange(t_span[0], t_span[1] + dt, dt)
    N_analytic = decay_analytic(t_analytic, N0, lam)
    print(f"Analytic solution computed: {len(t_analytic)} points")

    results = { "euler": (t_euler, N_euler), 
           "rk4": (t_rk4, N_rk4), 
           "scipy": (t_scipy, N_scipy),
           "analytic": (t_analytic, N_analytic),
      }
    return results

    # Plotting results
def plot_results(results):
    plt.figure(figsize=(12, 8))

    #unpack results
    t_euler, N_euler = results["euler"]
    t_rk4, N_rk4 = results["rk4"]
    t_scipy, N_scipy = results["scipy"]
    t_analytic, N_analytic = results["analytic"]

    plt.plot(t_euler,N_euler, label='Euler Method', linestyle='--',color="red")
    plt.plot(t_rk4, N_rk4, label='RK4 Method', linestyle='-.',color="Blue")
    plt.plot(t_scipy, N_scipy, label='SciPy solve_ivp', linestyle=':',color="Green")
    plt.plot(t_analytic, N_analytic, label='Analytic Solution',color='black')
    
    plt.title('Decay Process Simulation')
    plt.xlabel('Time')
    plt.ylabel('Quantity N(t)')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_errors(results):
    """Calculate and print errors for each method."""
    t_analytic, N_analytic = results["analytic"]
    
    print("\nError Analysis:")
    print("-" * 40)
    
    for method in ["euler", "rk4", "scipy"]:
        t_method, N_method = results[method]
        
        # Interpolate analytic solution to match method's time points
        N_analytic_interp = np.interp(t_method, t_analytic, N_analytic)
        
        # Calculate errors
        max_error = np.max(np.abs(N_method - N_analytic_interp))
        rmse = np.sqrt(np.mean((N_method - N_analytic_interp)**2))
        
        print(f"{method.upper():>6}: Max Error = {max_error:.2e}, RMSE = {rmse:.2e}")

def validate_euler_convergence(N0, lam, t_span):
    """Validate Euler method by testing convergence with different step sizes"""
    print("\n" + "="*50)
    print("Euler Method Convergence Validation")
    print("="*50)
    
    dt_values = [1.0, 0.5, 0.1, 0.05]
    errors = []
    
    def f(t, N):
        return decay_process(t, N, lam)
    
    for dt in dt_values:
        t_euler, N_euler = euler_method(f, t_span, N0, dt)
        
        # Compare with analytic solution at final time
        N_analytic_final = decay_analytic(t_span[1], N0, lam)
        error = abs(N_euler[-1] - N_analytic_final)
        errors.append(error)
        
        print(f"dt = {dt:.2f}: Final N_euler = {N_euler[-1]:.4f}, "
              f"N_analytic = {N_analytic_final:.4f}, Error = {error:.4f}")
    
    # Check if error decreases with smaller dt
    print("\nConvergence check:")
    for i in range(1, len(errors)):
        ratio = errors[i-1] / errors[i] if errors[i] != 0 else float('inf')
        print(f"Error ratio (dt={dt_values[i-1]:.2f}/dt={dt_values[i]:.2f}) = {ratio:.2f}")

def calculate_global_truncation_errors(N0, lam, t_span, dt_values):
    """Calculate global truncation errors for different methods and step sizes"""
    
    print("\n" + "="*60)
    print("GLOBAL TRUNCATION ERROR ANALYSIS")
    print("="*60)
    
    # Store errors for each method
    errors_euler = []
    errors_rk4 = []
    errors_scipy = []
    
    def f(t, N):
        return decay_process(t, N, lam)
    
    for dt in dt_values:
        print(f"\nAnalyzing dt = {dt:.3f}")
        
        # Run simulation with current dt
        results = run_decay(N0, lam, dt, t_span)
        
        # Get analytic solution at final time
        t_final = t_span[1]
        N_analytic_final = decay_analytic(t_final, N0, lam)
        
        # Calculate global truncation error at final time
        t_euler, N_euler = results["euler"]
        t_rk4, N_rk4 = results["rk4"]
        t_scipy, N_scipy = results["scipy"]
        
        error_euler = abs(N_euler[-1] - N_analytic_final)
        error_rk4 = abs(N_rk4[-1] - N_analytic_final)
        error_scipy = abs(N_scipy[-1] - N_analytic_final)
        
        errors_euler.append(error_euler)
        errors_rk4.append(error_rk4)
        errors_scipy.append(error_scipy)
        
        print(f"  Euler:  {error_euler:.6f}")
        print(f"  RK4:    {error_rk4:.6f}")
        print(f"  SciPy:  {error_scipy:.6f}")
    
    return dt_values, errors_euler, errors_rk4, errors_scipy

def plot_global_truncation_errors(dt_values, errors_euler, errors_rk4, errors_scipy):
    """Plot global truncation errors vs step size"""
    
    plt.figure(figsize=(12, 8))
    
    # Convert to arrays for calculations
    dt_array = np.array(dt_values)
    errors_euler_array = np.array(errors_euler)
    errors_rk4_array = np.array(errors_rk4)
    errors_scipy_array = np.array(errors_scipy)
    
    # Plot actual errors
    plt.loglog(dt_array, errors_euler_array, 'ro-', label='Euler Error', linewidth=2, markersize=8)
    plt.loglog(dt_array, errors_rk4_array, 'bs-', label='RK4 Error', linewidth=2, markersize=8)
    plt.loglog(dt_array, errors_scipy_array, 'g^-', label='SciPy Error', linewidth=2, markersize=8)
    
    # Plot theoretical error bounds
    # Euler: O(dt), RK4: O(dt^4), SciPy: O(dt^4) or better
    scale_euler = errors_euler_array[0] / dt_array[0]
    scale_rk4 = errors_rk4_array[0] / (dt_array[0]**4)
    
    plt.loglog(dt_array, scale_euler * dt_array, 'r--', 
               label='Theoretical O(dt) - Euler', alpha=0.7)
    plt.loglog(dt_array, scale_rk4 * dt_array**4, 'b--', 
               label='Theoretical O(dt⁴) - RK4', alpha=0.7)
    
    plt.xlabel('Time Step Size (dt)', fontsize=12)
    plt.ylabel('Global Truncation Error at Final Time', fontsize=12)
    plt.title('Global Truncation Error vs Step Size', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', alpha=0.3)
    
    # Add convergence rate calculations
    plt.text(0.02, 0.02, 
             f'Convergence Rates:\nEuler: O(dt)\nRK4: O(dt⁴)\nSciPy: O(dt⁴) or adaptive',
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    return dt_array, errors_euler_array, errors_rk4_array, errors_scipy_array

def calculate_convergence_rates(dt_values, errors_euler, errors_rk4, errors_scipy):
    """Calculate numerical convergence rates"""
    
    print("\n" + "-"*50)
    print("CONVERGENCE RATE CALCULATIONS")
    print("-"*50)
    
    dt_array = np.array(dt_values)
    errors_euler_array = np.array(errors_euler)
    errors_rk4_array = np.array(errors_rk4)
    
    print(f"{'Method':<10} {'dt1->dt2':<12} {'Rate':<8} {'Theoretical':<12}")
    print("-" * 50)
    
    # Calculate Euler convergence rates
    for i in range(1, len(dt_array)):
        if errors_euler_array[i] > 0 and errors_euler_array[i-1] > 0:
            rate_euler = np.log(errors_euler_array[i-1] / errors_euler_array[i]) / np.log(dt_array[i-1] / dt_array[i])
            print(f"{'Euler':<10} {dt_array[i-1]:.3f}->{dt_array[i]:.3f} {rate_euler:>6.3f}   {1.0:>8.1f}")
    
    print("-" * 30)
    
    # Calculate RK4 convergence rates
    for i in range(1, len(dt_array)):
        if errors_rk4_array[i] > 0 and errors_rk4_array[i-1] > 0:
            rate_rk4 = np.log(errors_rk4_array[i-1] / errors_rk4_array[i]) / np.log(dt_array[i-1] / dt_array[i])
            print(f"{'RK4':<10} {dt_array[i-1]:.3f}->{dt_array[i]:.3f} {rate_rk4:>6.3f}   {4.0:>8.1f}")

def get_user_input():
    """Get user from user input for simulation parameters."""
    print("Enter simulation parameters:")

    try:
        N0 =float(input("Initial quantity (N0,default 100): "))
        lam = float(input("Decay constant (λ, default 0.1): "))
        T = float(input("Final time T (default 50): "))
        dt = float(input("Time step size (dt, default 0.1): "))
    except ValueError:
        print("Invalid input. Using default values.")
        N0 = 100.0
        lam = 0.1
        T = 50.0
        dt = 0.1
    return N0, lam, dt, T

if __name__ == "__main__":
    # Set default values
    run_global_analysis = False
    
    if len(sys.argv) > 1:
        # Command line arguments
        parser = argparse.ArgumentParser(description='Simulate decay process using different ODE solvers.')
        parser.add_argument('--N0', type=float, default=1000.0, help='Initial quantity')
        parser.add_argument('--lam', type=float, default=0.3, help='Decay constant')
        parser.add_argument('--dt', type=float, default=2.0, help='Time step size')
        parser.add_argument('--T', type=float, default=15.0, help='Final time')
        parser.add_argument('--error-analysis', action='store_true', help='Run global truncation error analysis')
        
        args = parser.parse_args()
        N0, lam, dt, T = args.N0, args.lam, args.dt, args.T
        run_global_analysis = args.error_analysis
    else:
        # Interactive input
        N0, lam, dt, T = get_user_input()
        # Ask about global analysis
        run_global_analysis = input("\nRun global error analysis? (y/n): ").lower().startswith('y')
    
    t_span = (0, T)

    print("=" * 50)
    print("Decay Process Simulation - Method Comparison")
    print("=" * 50)
    print(f"Initial quantity (N0): {N0}")
    print(f"Decay constant (λ): {lam}")
    print(f"Time step (dt): {dt}")
    print(f"Final time (T): {T}")
    
    # Run single simulation
    print("\nRunning single simulation with current parameters...")
    results = run_decay(N0, lam, dt, t_span)
    
    # Plot comparison graphs
    print("\nGenerating main comparison graph...")
    plot_results(results)
    calculate_errors(results)
    
    # Run global truncation error analysis if requested
    if run_global_analysis:
        print("\n" + "="*60)
        print("STARTING GLOBAL ERROR ANALYSIS")
        print("="*60)
        
        dt_values = [2.0, 1.0, 0.5, 0.25, 0.125]  # Different step sizes to test
        
        dt_vals, errors_euler, errors_rk4, errors_scipy = calculate_global_truncation_errors(
            N0, lam, t_span, dt_values)
        
        plot_global_truncation_errors(dt_vals, errors_euler, errors_rk4, errors_scipy)
        calculate_convergence_rates(dt_vals, errors_euler, errors_rk4, errors_scipy)
    else:
        print("\nSkipping global error analysis")
    
    # Validate Euler method convergence
    validate_euler_convergence(N0, lam, t_span)
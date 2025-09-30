import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.integrate import cumulative_trapezoid

from Integeral_methods import riemann_sum, trapezoidal_rule, simpsons_rule
from physics_problem import spring_force, work_done_by_spring 

def run_spring_work(a, b, k, n):
    # Define the spring force function
    def f(x):
        return spring_force(k, x)
    
    # Numerical integration using different methods
    riemann_result = riemann_sum(f, a, b, n)
    trapezoidal_result = trapezoidal_rule(f, a, b, n)
    simpsons_result = simpsons_rule(f, a, b, n)

    # SciPy integration for comparison
    x = np.linspace(a, b, n + 1)
    y = f(x)
    trap_scipy_result = integrate.trapezoid(y, x)
    simp_scipy_result = integrate.simpson(y, x)
    
    # Analytical result
    analytical_result = work_done_by_spring(k, a, b)
    
    results = {
        "Riemann Sum": riemann_result,
        "Trapezoidal Rule": trapezoidal_result,
        "Simpson's Rule": simpsons_result,
        "SciPy Trapezoidal": trap_scipy_result,
        "SciPy Simpson": simp_scipy_result,
        "Analytical": analytical_result
    }
    return results

# Print results
def print_results(results):   
    # print all results in a table format
    print("\nWork done by spring:")
    print("-" * 40)
    for method, value in results.items():
        print(f"{method:20s}: {value:.6f}")

def plot_integrand(a, b, k):
    # Plot the spring force curve and shaded work area.
    x = np.linspace(a, b, 200)
    y = spring_force(k, x)
    plt.figure(figsize=(7, 5))
    plt.plot(x, y, label="Spring force F(x) = -kx", color="blue")
    plt.fill_between(x, 0, y, alpha=0.2, color="blue")
    plt.xlabel("Displacement x")
    plt.ylabel("Force F(x)")
    plt.title("Work done by spring (area under curve)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_comparison(a, b, k, n):
    # Comparison between trapezoidal method and analytic solution.
    x_vals = np.linspace(a, b, n + 1)
    f_vals = spring_force(k, x_vals)

    # Cumulative trapezoidal integration
    cum_trapz = cumulative_trapezoid(f_vals, x_vals, initial=0)

    plt.figure(figsize=(7, 5))
    plt.plot(x_vals, cum_trapz, label="Trapezoidal cumulative", linestyle="--")
    plt.plot(x_vals, 0.5 * k * (x_vals**2 - a**2), label="Analytical", color="black")
    plt.xlabel("Displacement x")
    plt.ylabel("Work W(x)")
    plt.title("Cumulative Work Done by Spring")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_error_convergence(a, b, k, max_n=2000):
    # Show error convergence of numerical methods vs analytic solution.
    
    analytic = work_done_by_spring(k, a, b)
    ns = np.logspace(1, np.log10(max_n), 20, dtype=int)
    errors_trap, errors_simp, errors_riemann = [], [], []

    for n in ns:
        def f(x):
            return spring_force(k, x)
        x = np.linspace(a, b, n + 1)
        y = f(x)

        trap_val = integrate.trapezoid(y, x)
        simp_val = integrate.simpson(y, x)
        riemann_val = riemann_sum(f, a, b, n)

        errors_trap.append(abs(trap_val - analytic))
        errors_simp.append(abs(simp_val - analytic))
        errors_riemann.append(abs(riemann_val - analytic))

    # Calculate theoretical truncation errors
    h_values = (b - a) / ns

    plt.figure(figsize=(10, 6))
    plt.loglog(ns, errors_riemann, '^-', label="Riemann Error", color='red')
    plt.loglog(ns, errors_trap, "o-", label="Trapezoidal Error", color="blue")
    plt.loglog(ns, errors_simp, "s-", label="Simpson Error", color="green")
    
    # Plot theoretical truncation bounds (dashed lines)
    scale_factor = max(errors_riemann) / ((b - a) / min(ns))  # Scale for visibility
    
    plt.loglog(ns, scale_factor * h_values, '--', 
               label="Theoretical O(h) - Riemann", color='red', alpha=0.7)
    plt.loglog(ns, scale_factor * h_values**2, '--', 
               label="Theoretical O(h²) - Trapezoidal", color='blue', alpha=0.7)
    plt.loglog(ns, scale_factor * h_values**4, '--', 
               label="Theoretical O(h⁴) - Simpson", color='green', alpha=0.7)
    
    plt.xlabel("Number of intervals (n)")
    plt.ylabel("Error Magnitude")
    plt.title("Numerical Integration Errors: Actual vs Theoretical Truncation Bounds")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work done by spring simulation")
    parser.add_argument("--a", type=float, default=None, help="Initial displacement")
    parser.add_argument("--b", type=float, default=None, help="Final displacement")
    parser.add_argument("--k", type=float, default=None, help="Spring constant")
    parser.add_argument("--n", type=int, default=100, help="Number of intervals")
    args = parser.parse_args()

    # Ask user for inputs if not given in command line
    if args.a is None:
        args.a = float(input("Enter initial displacement a: "))
    if args.b is None:
        args.b = float(input("Enter final displacement b: "))
    if args.k is None:
        args.k = float(input("Enter spring constant k: "))

    results = run_spring_work(args.a, args.b, args.k, args.n)
    print_results(results)

    # Plots
    plot_integrand(args.a, args.b, args.k)
    plot_comparison(args.a, args.b, args.k, args.n)
    plot_error_convergence(args.a, args.b, args.k)




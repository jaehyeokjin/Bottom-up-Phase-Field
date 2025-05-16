import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sympy as sp

def custom_polynomial(x, *coeffs):
    """Custom polynomial model that inherently satisfies f(0)=f(1)=0 and f'(0)=f'(1)=0."""
    n = len(coeffs)  # Determine the degree based on number of coefficients
    P = sum([coeffs[i] * x**i for i in range(n)])
    return x**2 * (1 - x)**2 * P

# Load data
data = np.loadtxt('smooth_g_f_profiles.dat')
x_data, f_data = data[:, 0], data[:, 1]

# Symbolic variable for sympy operations
x_sym = sp.symbols('x')

# Plot the original data
plt.figure(figsize=(12, 8))
plt.plot(x_data, f_data, 'ko', label='Original Data', markersize=4)

colors = plt.cm.viridis(np.linspace(0, 1, 12))  # Colormap for different polynomial degrees
degree_P = 2
initial_coeffs = np.zeros(degree_P + 1)
params, _ = curve_fit(custom_polynomial, x_data, f_data, p0=initial_coeffs)
# Create the polynomial and derivatives/integrals
polynomial = sum([params[i] * x_sym**i for i in range(len(params))]) * x_sym**2 * (1 - x_sym)**2
polynomial_prime = sp.diff(polynomial, x_sym)
integral_f = sp.integrate(polynomial, (x_sym, 0, 1))
p_prime_x = polynomial / integral_f
p_x = sp.integrate(polynomial, (x_sym, 0, x_sym)) / integral_f

# Print results for each polynomial degree
print(f"Fitted polynomial f(x) for degree {degree_P + 4}:")
print(sp.simplify(polynomial))
print(f"Derivative f'(x) for degree {degree_P + 4}:")
print(sp.simplify(polynomial_prime))
print(f"Normalized polynomial p(x) for degree {degree_P + 4}:")
print(sp.simplify(p_x))
print(f"Derivative p'(x) for degree {degree_P + 4}:")
print(sp.simplify(p_prime_x))

# Plot the fitted polynomial
func = sp.lambdify(x_sym, polynomial, 'numpy')
plt.plot(x_data, func(x_data), '-', color=colors[degree_P-4], label=f'Degree {degree_P + 4}')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Fitted Polynomial Functions of Various Degrees')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()


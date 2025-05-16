import numpy as np
from scipy.integrate import simps

# Load the data (assuming the first column is x and the second column is g'(x))
data = np.loadtxt('q_phi.dat')
x_values = data[:, 0]
g_prime_values = data[:, 1]

# Ensure x_values starts at 0 for accurate integration from 0
if x_values[0] != 0:
    x_values = np.insert(x_values, 0, 0)
    g_prime_values = np.insert(g_prime_values, 0, 0)

# Calculate g(x) by integrating g'(x)
g_values = np.array([simps(g_prime_values[:i+1], x_values[:i+1]) for i in range(len(x_values))])

# Calculate the denominator for f'(x), which is the integral of g(x) from 0 to 1
denom_f_prime = simps(g_values, x_values)

# Calculate f'(x) = g(x) / integral of g(x) from 0 to 1
f_prime_values = g_values / denom_f_prime

# Calculate f(x) by integrating f'(x)
f_values = np.array([simps(f_prime_values[:i+1], x_values[:i+1]) for i in range(len(x_values))])

# Save the results x:g(x):f(x)
np.savetxt('g_f_profiles.dat', np.column_stack((x_values, g_values, f_values)), fmt='%f')

print("The calculation is completed and saved to 'g_f_profiles.dat'.")


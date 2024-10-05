import numpy as np
from scipy.integrate import simps

# Load g'(phi) data from "q_phi.dat"
data = np.loadtxt('q_phi.dat')
x_values = data[:, 0]  # x or phi
g_prime_values = data[:, 1]  # g'(phi)

# Ensure the starting point is 0 for proper integration
if x_values[0] != 0:
    x_values = np.insert(x_values, 0, 0)
    g_prime_values = np.insert(g_prime_values, 0, 0)

# Calculate g(x) by integrating g'(phi) from 0 to x
g_values = np.array([simps(g_prime_values[:i+1], x_values[:i+1]) for i in range(len(x_values))])

# Calculate the integral of g(x) from 0 to 1
integral_g_0_1 = simps(g_values, x_values)

# Calculate f'(x) = g(x) / integral of g(x) from 0 to 1
f_prime_values = g_values / integral_g_0_1

# Save the results to "p_phi.dat" with format x:f'(x)
np.savetxt('p_phi.dat', np.column_stack((x_values, f_prime_values)), fmt='%f')

print("The calculation of f'(x) based on g'(phi) is completed and saved to 'p_phi.dat'.")



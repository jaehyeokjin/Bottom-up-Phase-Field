import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as uvs
from scipy.integrate import simps

# Load the data
data = np.loadtxt('q_phi.dat')
x_values = data[:, 0]
g_prime_values = data[:, 1]

# Create a spline representation of g'(x)
g_prime_spline = uvs(x_values, g_prime_values, k=3)

# Calculate g(x) by integrating the spline approximation of g'(x)
# We need a fine grid for integration to improve accuracy
fine_x = np.linspace(x_values[0], x_values[-1], 1000)
g_values = np.array([g_prime_spline.integral(x_values[0], xi) for xi in fine_x])

# Calculate the integral of g(x) from 0 to 1 for normalization
# Assumption: The range of x includes 0 to 1, adjust if needed
integral_g_0_1 = simps(g_values[fine_x <= 1], fine_x[fine_x <= 1])

# Calculate f'(x) = g(x) / integral of g(x) from 0 to 1
f_prime_values = g_values / integral_g_0_1

# Calculate f(x) by integrating f'(x)
f_values = np.array([simps(f_prime_values[:i+1], fine_x[:i+1]) for i in range(len(fine_x))])

# Save the results x:g(x):f(x)
np.savetxt('smooth_g_f_profiles.dat', np.column_stack((fine_x, g_values, f_values)), fmt='%f')

# Calculate g'(x) using the spline at fine_x points
evaluated_g_prime = g_prime_spline(fine_x)

# Save the derivative profiles x:g'(x):f'(x)
np.savetxt('smooth_derivative_g_f_profiles.dat', np.column_stack((fine_x, evaluated_g_prime, f_prime_values)), fmt='%f')

print("The calculations are completed and saved to 'g_f_profiles.dat' and 'derivative_g_f_profiles.dat'.")


import numpy as np
import scipy.optimize as opt

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define parameters
L = 0.32        # Length of the domain (m)
nx = 200        # Number of spatial nodes
dx = L / nx    # Spatial step

T_inf = 25    # Ambient temperature (K)
q_left = 44.53   # Heat flux at left boundary (W/m^2)
q_right = 378.36 # Heat flux at right boundary (W/m^2)

# Function to solve steady-state temperature profile
def solve_temperature(h, k):
    A = np.zeros((nx, nx))
    b = np.zeros(nx)
    
    for i in range(1, nx-1):
        A[i, i-1] = k / dx**2
        A[i, i] = -2 * k / dx**2 - h / k
        A[i, i+1] = k / dx**2
        b[i] = -h * T_inf / k
    
    # Left boundary (constant heat flux)
    A[0, 0] = -k / dx
    A[0, 1] = k / dx
    b[0] = -q_left
    
    # Right boundary (constant heat flux)
    A[-1, -1] = -k / dx
    A[-1, -2] = k / dx
    b[-1] = -q_right
    
    return np.linalg.solve(A, b)

# Function to fit h and k to measured data
def fit_model(params, x_measured, T_measured):
    h, k = params
    x = np.linspace(0, L, nx)
    T_model = solve_temperature(h, k)
    return np.interp(x_measured, x, T_model) - T_measured

# Example measured data (user input needed)
x_measured = np.array([10, 85.0, 160.0, 235.0, 310.0])/1000
T_measured = np.array([25.482772764235765, 25.547115704295706, 25.793495117882117, 26.42292883916084, 27.043057390609388])

# Initial guesses for h and k
initial_guess = [5, 20]

# Curve fitting to find optimal h and k
optimal_params, _ = curve_fit(lambda x, h, k: fit_model((h, k), x_measured, T_measured), x_measured, np.zeros_like(x_measured), p0=initial_guess)
h_opt, k_opt = optimal_params

# Solve using optimized parameters
T_opt = solve_temperature(h_opt, k_opt)

# Plot results
x = np.linspace(0, L, nx)
plt.plot(x, T_opt, label='Fitted Model')
plt.scatter(x_measured, T_measured, color='red', label='Measured Data')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (K)')
plt.title('1D Steady-State Heat Conduction with Fitted Parameters')
plt.legend()
plt.show()

print(f'Optimized h: {h_opt:.2f} W/m^2K')
print(f'Optimized k: {k_opt:.2f} W/mK')

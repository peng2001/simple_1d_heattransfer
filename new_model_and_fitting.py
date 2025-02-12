import numpy as np
import scipy.optimize as opt

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define parameters
L = 0.354        # Length of the domain (m)
nx = 200        # Number of spatial nodes
dx = L / nx    # Spatial step

T_inf = 23.2   # Ambient temperature (K)
q_left = 586.19  # Heat flux at left boundary (W/m^2)
q_right = 785.73 # Heat flux at right boundary (W/m^2)
x_measured = np.linspace((354-320)/2+10, (354-320)/2+310, 5)/1000
T_measured = np.array([32.353269382617384, 31.592185749250753, 31.53997257342657, 32.630572063936064, 34.004256928071925])
# Function to solve steady-state temperature profile
def solve_temperature(h, k):
    A = np.zeros((nx, nx))
    b = np.zeros(nx)
    
    for i in range(1, nx-1):
        A[i, i-1] = k / dx**2
        A[i, i] = -2 * k / dx**2 - h / k
        A[i, i+1] = k / dx**2
        b[i] = -h * T_inf / k
    
    # Left boundary (constant heat flux + convection)
    A[0, 0] = -k / dx
    A[0, 1] = k / dx
    b[0] = -q_left
    
    # Right boundary (constant heat flux + convection)
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

# Example measured data
# x_measured = np.linspace(10, 310, 5)/1000

# Initial guesses for h and k
initial_guess = [90000, 20]

# Curve fitting to find optimal h and k
optimal_params, _ = curve_fit(lambda x, h, k: fit_model((h, k), x_measured, T_measured), x_measured, np.zeros_like(x_measured), p0=initial_guess, maxfev=10000)
h_opt, k_opt = optimal_params
# h_fixed = -1000  # Set h to a fixed value
# optimal_params, _ = curve_fit(lambda x, k: fit_model((h_fixed, k), x_measured, T_measured), x_measured, np.zeros_like(x_measured), p0=[initial_guess[1]], maxfev=10000)
# h_opt = h_fixed
# k_opt = optimal_params[0]
# h_opt = 30000
# k_opt = 25
# Solve using optimized parameters
T_opt = solve_temperature(h_opt, k_opt)

# Plot results
x = np.linspace(0, L, nx)
plt.plot(x, T_opt, label='Fitted Model')
plt.scatter(x_measured, T_measured, color='red', label='Measured Data')
# plt.scatter([0,0.354], [20,25], color='orange', label='Tabs')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (K)')
plt.title('1D Steady-State Heat Conduction with Fitted Parameters')
plt.legend()
plt.show()

print(f'Optimized h: {h_opt:.2f} W/m^2K')
print(f'Optimized k: {k_opt:.2f} W/mK')



# def solve_temperature(h, k, T_inf):
#     A = np.zeros((nx, nx))
#     b = np.zeros(nx)
    
#     for i in range(1, nx-1):
#         A[i, i-1] = k / dx**2
#         A[i, i] = -2 * k / dx**2 - h / k
#         A[i, i+1] = k / dx**2
#         b[i] = -h * T_inf / k
    
#     # Left boundary (constant heat flux + convection)
#     A[0, 0] = -k / dx
#     A[0, 1] = k / dx
#     b[0] = -q_left
    
#     # Right boundary (constant heat flux + convection)
#     A[-1, -1] = -k / dx
#     A[-1, -2] = k / dx
#     b[-1] = -q_right
    
#     return np.linalg.solve(A, b)

# # Function to fit h, k, and T_inf to measured data
# def fit_model(params, x_measured, T_measured):
#     h, k, T_inf = params
#     x = np.linspace(0, L, nx)
#     T_model = solve_temperature(h, k, T_inf)
#     return np.interp(x_measured, x, T_model) - T_measured

# # Example measured data (user input needed)

# # Initial guesses for h, k, and T_inf
# initial_guess = [10000, 21, 23]

# # Curve fitting to find optimal h, k, and T_inf
# bounds_lower = [1000, 10, 20]  # Min values for h, k, and T_inf
# bounds_upper = [100000, 50, 40]  # Max values for h, k, and T_inf

# optimal_params, _ = curve_fit(lambda x, h, k, T_inf: fit_model((h, k, T_inf), x_measured, T_measured), 
#                               x_measured, np.zeros_like(x_measured), p0=initial_guess,
#                               bounds=(bounds_lower, bounds_upper), maxfev = 100000)
# h_opt, k_opt, T_inf_opt = optimal_params
# # Solve using optimized parameters
# T_opt = solve_temperature(h_opt, k_opt, T_inf_opt)

# # Plot results
# x = np.linspace(0, L, nx)
# plt.plot(x, T_opt, label='Fitted Model')
# plt.scatter(x_measured, T_measured, color='red', label='Measured Data')
# plt.xlabel('Position (m)')
# plt.ylabel('Temperature (K)')
# plt.title('1D Steady-State Heat Conduction with Fitted Parameters')
# plt.legend()
# plt.show()

# print(f'Optimized h: {h_opt:.2f} W/m^2K')
# print(f'Optimized k: {k_opt:.2f} W/mK')
# print(f'Optimized T_inf: {T_inf_opt:.2f} K')
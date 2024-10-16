import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0, k, elementary_charge, m_e, pi, h
from scipy.sparse import diags, kron
from scipy.sparse.linalg import spsolve

# Constants Parameters
temperature = 4  
q = elementary_charge

# Simulation Parameters
LENGTH = 22e-9
WIDTH = 40e-9
INSULATOR = 1e-9
DOPING = 1.0e21
XGRID = 1e-9
YGRID = 0.25e-9

# Grid
nx, ny = int(LENGTH / XGRID), int(WIDTH / YGRID)

# Material permittivities
epsilon_r = np.ones((ny, nx)) * 11.7 * epsilon_0  # Silicon
epsilon_r[:int(INSULATOR / YGRID), :] = 3.9 * epsilon_0  # SiO2 top
epsilon_r[-int(INSULATOR / YGRID):, :] = 3.9 * epsilon_0  # SiO2 bottom

def adjust_bandgap_for_temperature(T):
    Eg_eV = 1.21 - 3.60e-4 * T  
    return Eg_eV * q  # Convert to Joules

def effective_density_of_states(m_eff, T):
    return 2 * (2 * pi * m_eff * k * T / (h ** 2)) ** (3 / 2)

def intrinsic_carrier_concentration(N_C, N_V, Eg, T):
    return np.sqrt(N_C * N_V) * np.exp(-Eg / (2 * k * T))

def calculate_rho(ny, nx, doping_conc, ni, quantum_dot_center, quantum_dot_radius, quantum_dot_doping):
    rho_matrix = np.ones((ny, nx)) * (doping_conc - ni)  # Background charge density
    y_center, x_center = quantum_dot_center
    y_radius, x_radius = quantum_dot_radius
    for i in range(max(0, y_center - y_radius), min(ny, y_center + y_radius)):
        for j in range(max(0, x_center - x_radius), min(nx, x_center + x_radius)):
            rho_matrix[i, j] += quantum_dot_doping  # Add charge density in the quantum dot region
    return rho_matrix * q  # Convert to charge per volume

def solve_poisson(rho, epsilon_r, dx, dy):
    ny, nx = epsilon_r.shape
    Lx = diags([1, -2, 1], [-1, 0, 1], shape=(nx, nx)) / dx**2
    Ly = diags([1, -2, 1], [-1, 0, 1], shape=(ny, ny)) / dy**2
    L = kron(Ly, diags([1]*nx)) + kron(diags([1]*ny), Lx)

    epsilon_flattened = epsilon_r.flatten()
    source_term = -rho.flatten() / epsilon_flattened
    V_flat = spsolve(L, source_term)

    return V_flat.reshape((ny, nx))

# def solve_poisson(rho, epsilon_r, dx, dy):
#     ny, nx = epsilon_r.shape
#     Lx = diags([1, -2, 1], [-1, 0, 1], shape=(nx, nx)) / dx**2
#     Ly = diags([1, -2, 1], [-1, 0, 1], shape=(ny, ny)) / dy**2
#     L = kron(Ly, diags([1]*nx)) + kron(diags([1]*ny), Lx)
#     L /= epsilon_r.flatten()
#     V_flat = spsolve(L, -rho.flatten())
#     return V_flat.reshape((ny, nx))




# Quantum dot characteristics
quantum_dot_center = (ny//2, nx//2)  
quantum_dot_radius = (int(5e-9 / YGRID), int(5e-9 / XGRID))  
quantum_dot_doping = 5.0e21  

# Bandgap energy and intrinsic concentration
Eg = adjust_bandgap_for_temperature(temperature)
m_eff_electron = 0.26 * m_e
m_eff_hole = 0.37 * m_e
N_C = effective_density_of_states(m_eff_electron, temperature)
N_V = effective_density_of_states(m_eff_hole, temperature)
ni = intrinsic_carrier_concentration(N_C, N_V, Eg, temperature)

# Calculate charge density for quantum dot
rho_T = calculate_rho(ny, nx, DOPING, ni, quantum_dot_center, quantum_dot_radius, quantum_dot_doping)

# Solve the Poisson equation
V_T = solve_poisson(rho_T, epsilon_r, XGRID, YGRID)

# Visualization
plt.imshow(V_T, extent=(0, LENGTH*1e9, 0, WIDTH*1e9), origin='lower', cmap='viridis')
plt.colorbar(label='Potential (V)')
# plt.title(f'Electrostatic Potential Distr. at {temperature}K')
plt.xlabel('x (nm)')
plt.ylabel('y (nm)')
plt.savefig('2DSemiQubit_Potat4K.png', dpi=300)
plt.show()

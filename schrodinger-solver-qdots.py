import numpy as np
from scipy.constants import hbar, m_e, pi, elementary_charge, epsilon_0
from scipy.sparse import kron, diags
from scipy.sparse.linalg import eigs

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

# Constants
m_eff = 0.26 * m_e  # Effective mass for Silicon
Lx = 22e-9  # Length of the quantum well in meters
Ly = 40e-9  # Width of the quantum well in meters
nx = int(Lx / 1e-9)  # Number of grid points in x
ny = int(Ly / 0.25e-9)  # Number of grid points in y
dx = Lx / (nx - 1)  # Grid spacing in x
dy = Ly / (ny - 1)  # Grid spacing in y

def schrodinger_solver_2D_with_materials():
    # Assuming V0 as potential energy in Si regions and a higher V1 in SiO2 regions
    V0 = 0  # Potential energy inside the silicon quantum well
    V1 = 100 * elementary_charge  # Arbitrary higher potential energy for SiO2 barrier
    
    # Create potential energy landscape
    V = np.ones((ny, nx)) * V0
    V[:int(INSULATOR / YGRID), :] = V1  # Apply higher potential for top SiO2
    V[-int(INSULATOR / YGRID):, :] = V1  # Apply higher potential for bottom SiO2
    
    # Convert to 1D array for compatibility with Hamiltonian construction
    V_flat = V.flatten()
    
    # Construct the Hamiltonian with the adjusted potential energy
    Tx = -(hbar**2 / (2 * m_eff)) * diags([1, -2, 1], [-1, 0, 1], shape=(nx, nx)) / dx**2
    Ty = -(hbar**2 / (2 * m_eff)) * diags([1, -2, 1], [-1, 0, 1], shape=(ny, ny)) / dy**2
    H = kron(Ty, diags([1]*nx)) + kron(diags([1]*ny), Tx) + diags(V_flat)
    
    # Solve for the eigenvalues and eigenvectors
    energies, wavefunctions = eigs(H, k=5, which='SM', return_eigenvectors=True)
    energies = np.real(energies)
    wavefunctions = np.real(wavefunctions)
    
    return energies, wavefunctions.reshape((ny, nx, -1)), nx, ny


def schrodinger_solver_2D():
    """Solve the 2D Schrödinger equation for a rectangular quantum well."""
    # Kinetic energy operator in x direction
    Tx = -(hbar**2 / (2 * m_eff)) * diags([1, -2, 1], [-1, 0, 1], shape=(nx, nx)) / dx**2
    # Kinetic energy operator in y direction
    Ty = -(hbar**2 / (2 * m_eff)) * diags([1, -2, 1], [-1, 0, 1], shape=(ny, ny)) / dy**2
    # Full Hamiltonian
    H = kron(Ty, diags([1]*nx)) + kron(diags([1]*ny), Tx)
    
    # Solve for the lowest eigenvalues and corresponding eigenvectors
    energies, wavefunctions = eigs(H, k=5, which='SM', return_eigenvectors=True)
    
    # Ensure real parts are taken (physical solutions)
    energies = np.real(energies)
    wavefunctions = np.real(wavefunctions)
    
    return energies, wavefunctions.reshape((ny, nx, -1)), nx, ny

energies, wavefunctions, nx, ny = schrodinger_solver_2D_with_materials()

# Visualization
import matplotlib.pyplot as plt

# Assuming energies and wavefunctions are already defined
# Define the dimensions of the plot grid
rows = 3
cols = 2
fig, axs = plt.subplots(rows, cols, figsize=(12, 18))  # Adjust figsize as needed

# Flatten the array of axes for easy indexing
axs_flat = axs.flatten()

for i in range(len(energies)):
    ax = axs_flat[i]
    im = ax.imshow(wavefunctions[:, :, i], extent=[0, Lx*1e9, 0, Ly*1e9], origin='lower', aspect='auto')
    ax.set_title(f'Energy level {i+1}: {energies[i]:.2e} J')
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    fig.colorbar(im, ax=ax)  # Add a colorbar for each subplot

# Hide any unused subplots
for j in range(i + 1, rows * cols):
    axs_flat[j].axis('off')

plt.tight_layout()  # Adjust subplots to fit into the figure area.
plt.savefig('Schro', dpi=300)
plt.show()


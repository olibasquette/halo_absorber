import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio

# Use LaTeX for all text in the plot
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "axes.formatter.use_mathtext": True,
    "mathtext.fontset": "cm",
})

# Script to visualise optical depth (tau) in a 3D space (axes are xHI, Mvir, z), with a colour scale indicating the value of tau at a given photon energy (e.g., 0.5 keV)

def visualise_tau(xHI, Mvir, z, tau):
    """
    Visualises the optical depth (tau) in a 3D space defined by xHI, Mvir, and z.
    
    Parameters:
    xHI : array-like
        Neutral hydrogen fraction values.
    Mvir : array-like
        Virial mass values.
    z : array-like
        Redshift values.
    tau : array-like
        Optical depth values corresponding to the (xHI, Mvir, z) points.
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a scatter plot
    sc = ax.scatter(xHI, np.log10(Mvir), z, c=np.log10(tau), cmap='Reds', marker='o')
    
    # Add color bar
    cbar = plt.colorbar(sc,pad=0.15)
    cbar.set_label(r'log $\tau$ at 0.5 keV',fontsize=13)
    
    # Set labels
    ax.set_xlabel('Halo xHI',fontsize=12)
    ax.set_ylabel(r'log $M_\mathrm{vir}$ ($M_\odot$)',fontsize=12)
    ax.set_zlabel(r'$z$',fontsize=14)

    plt.savefig('./tau_3D_visualisation.pdf', dpi=300)
    plt.show()

# Import data

tau_data = sio.loadmat('halo_tau_grid.mat')
freq_bins = tau_data['freq_bins'].flatten()
redshift_bins = tau_data['redshift_bins'].flatten()
halo_masses_Msun = tau_data['halo_masses_Msun'].flatten()
xHI_bins = tau_data['xHI_bins'].flatten()
tau_arr = tau_data['tau_arr']

# For each (xHI, Mvir, z), find the value of tau at 0.5 keV
target_energy_keV = 0.5
target_energy_Hz = target_energy_keV * 1000 / 4.135667696e-15  # Convert keV to Hz
energy_index = np.argmin(np.abs(freq_bins - target_energy_Hz))
tau_at_target_energy = tau_arr[energy_index, :, :, :]

# Now plot

XHI, Mvir, Z = np.meshgrid(xHI_bins, halo_masses_Msun, redshift_bins, indexing='ij')
visualise_tau(XHI.flatten(), Mvir.flatten(), Z.flatten(), tau_at_target_energy.flatten())

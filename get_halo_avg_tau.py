# Script to calculate average tau at each redshift and frequency by averaging over the halo mass function

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from astropy import units as u

# Load HMF from file

data_HMF = sio.loadmat("./HMF_grid.mat")
halo_masses_Msun = data_HMF["halo_masses"].squeeze()
z_bins = data_HMF["z_grid"].squeeze()
hmf = data_HMF["HMF_grid"].squeeze() #shape (n_redshifts, n_halo_masses)

# Plot HMF against halo mass for different redshifts as sanity check (with colorbar for redshift)
plt.figure(figsize=(8,6))
cols = []
for i in range(len(z_bins)):
    z = z_bins[i]
    col = plt.cm.jet((z - np.min(z_bins))/ (np.max(z_bins) - np.min(z_bins)))
    cols.append(col)
    plt.plot(halo_masses_Msun, hmf[i,:], '-', color=col)
norm = plt.Normalize(vmin=np.min(z_bins), vmax=np.max(z_bins))
sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
sm.set_array(z_bins)
ax = plt.gca()
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label(r'Redshift z')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_\mathrm{vir}$ (Msun)')
plt.ylabel(r'$dn/dM$ (Mpc$^{-3}$ Msun$^{-1}$)')
plt.savefig('HMF_vs_Mvir.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Load tau grid from file

data_tau = sio.loadmat("./halo_tau_grid.mat")
tau_arr = data_tau["tau_arr"]  # shape (n_freqs, n_xHI, n_halo_masses, n_redshifts)
freq_bins = data_tau["freq_bins"].squeeze()  # in Hz
redshift_bins = data_tau["redshift_bins"].squeeze()
halo_masses_Msun = data_tau["halo_masses_Msun"].squeeze()
xHI_bins = data_tau["xHI_bins"].squeeze()

# First average tau over xHI (dependence is weak)

tau_avg_xHI = np.average(tau_arr, axis=1)  # shape (n_freqs, n_halo_masses, n_redshifts)

# Now average over halo mass function at each redshift and frequency
# <tau>(nu,z) = int tau(nu, Mvir, z) * dn/dm(Mvir,z) dMvir / int dn/dm(Mvir,z) dMvir
# HMF grid and tau grid already have same redshift bins and halo mass bins

tau_avg_mass = np.zeros((len(freq_bins), len(redshift_bins)))
for i in range(len(redshift_bins)):
    hmf_z = hmf[i,:]  # shape (n_halo_masses,)
    tau_z = tau_avg_xHI[:,:,i]  # shape (n_freqs, n_halo_masses)
    # Average over halo mass function
    print(f'shape of hmf_z: {hmf_z.shape}, shape of tau_z: {tau_z.shape}')
    for j in range(len(freq_bins)):
        tau_avg_mass[j,i] = np.trapz(tau_z[j,:] * hmf_z * halo_masses_Msun, x=np.log(halo_masses_Msun)) / np.trapz(hmf_z * halo_masses_Msun, x=np.log(halo_masses_Msun))

# Plot <tau> vs z

freq_plot = 0.1 * 1e3 * 2.417990504024e17  # convert keV to Hz
freq_index = np.argmin(np.abs(freq_bins - freq_plot))
plt.figure(figsize=(8,6))
plt.plot(redshift_bins, tau_avg_mass[freq_index,:], marker='o')
plt.xlabel('Redshift z')
plt.ylabel(r'$\langle\tau\rangle$')
plt.yscale('log')
plt.title(f'Average tau vs z at {freq_plot/(1e3*2.417990504024e17):.1f} keV')
plt.savefig('average_tau_vs_z.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Plot <tau> vs frequency (multiple lines for different redshifts with colorbar)

cols = []
for i in range(len(redshift_bins)):
    z = redshift_bins[i]
    col = plt.cm.jet((z - np.min(redshift_bins))/ (np.max(redshift_bins) - np.min(redshift_bins)))
    cols.append(col)

plt.figure(figsize=(8,6))
for i in range(len(redshift_bins)):
    plt.plot(freq_bins/(1e3*2.417990504024e17), tau_avg_mass[:,i], marker='o', color=cols[i])
plt.xlabel('Photon Energy (keV)')
plt.ylabel(r'$\langle\tau\rangle$')
ax = plt.gca()
norm = plt.Normalize(vmin=np.min(redshift_bins), vmax=np.max(redshift_bins))
sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
sm.set_array(redshift_bins)
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label(r'Redshift z')
plt.xscale('log')
plt.yscale('log')
#plt.xlim(1e-2, 1e2)
plt.ylim(1e-1,1e1)
plt.savefig('average_tau_vs_frequency.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Save the average tau grid to a file for later use
sio.savemat("halo_avg_tau_grid.mat", {"tau_avg_mass": tau_avg_mass,
                                     "freq_bins": freq_bins,
                                     "redshift_bins": redshift_bins})


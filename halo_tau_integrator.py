#%%

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

HZ_TO_EV = 4.135667696e-15  # eV/Hz

n_halo_masses = 50
n_xHIs = 10
#redshift_bins_coarse = np.arange(15,51,1) # integer spaced from 15 to 50
#redshift_bins_fine = np.arange(6,15,0.1) # spaced by 0.1 from 6 to 14.9
#redshift_bins_wide = np.concatenate((redshift_bins_fine, redshift_bins_coarse)) # all 21cmSPACE redshifts
redshift_bins_wide = np.arange(6,51,1) # integer spaced from 6 to 50
n_redshifts = len(redshift_bins_wide)
fine_freqs = np.genfromtxt("./fine_freqs.txt")
freq_bin_indices = [100, 200, 300, 350, 360, 364, 367, 370, 373, 376,
                    379, 382, 385, 388, 391, 394, 397, 400, 403, 406,
                    409, 412, 415, 418, 421, 424, 427, 430, 433, 436,
                    439, 442, 445, 448, 451, 454, 457, 460, 463, 466,
                    469, 472, 475, 478, 481, 485, 490, 500, 600, 700]
coarse_freqs = fine_freqs[freq_bin_indices]
halo_masses_Msun = np.logspace(3,13,n_halo_masses) # Msun
#xHI_bins = np.concatenate(([0], np.logspace(-5,0,n_xHIs-2), [1]))  # neutral fraction in haloes
xHI_bins = np.linspace(0,1,n_xHIs)  # neutral fraction in haloes


def get_tau_grid(integrand_arr, radii_arr):

    # integrate over r to get tau for all nu, xHI, Mvir, z
    tau_arr = np.trapz(integrand_arr*radii_arr[None,None,None,None,:], np.log10(radii_arr[None,None,None,None,:]), axis=-1)  # dimensionless
    tau_arr = tau_arr.squeeze()
    print(f"tau array shape: {tau_arr.shape}")
    return tau_arr

integrand_arr = np.load("integrand_arr.npy")
radii_arr = np.load("radii_arr.npy")

tau_arr = get_tau_grid(integrand_arr, radii_arr)

#%%

# test dependence on xHI by plotting slices with different colours base on xHI

plt.figure()
Mvir_val = halo_masses_Msun[25]
z_val = redshift_bins_wide[25]

cols = []
for i in range(len(xHI_bins)):
    xHI = xHI_bins[i]
    col = cm.jet((xHI - np.min(xHI_bins))/ (np.max(xHI_bins) - np.min(xHI_bins)))
    cols.append(col)

for i in range(0,len(xHI_bins)):
    tau_slice = tau_arr[:,i,25,25]  # select mid halo mass and redshift
    #plt.plot(coarse_freqs*HZ_TO_EV/1000, np.exp(-1*tau_slice), "-", label=rf'$xHI_\mathrm{{halo}} = {xHI_bins[i]:.2e}$') # plot in keV
    plt.plot(coarse_freqs*HZ_TO_EV/1000, np.exp(-1*tau_slice), "-", color=cols[i]) # plot in keV
plt.xlabel('Photon Energy (keV)')
plt.ylabel(r'Attenuation $e^{-\tau}$')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-4,1)
plt.xlim(1e-1,1e1)
norm = Normalize(vmin=np.min(xHI_bins), vmax=np.max(xHI_bins))
sm = cm.ScalarMappable(cmap=cm.jet, norm=norm)
sm.set_array(xHI_bins)
ax = plt.gca()
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label(r'$xHI_\mathrm{halo}$')
plt.title(f"Mvir = {Mvir_val:.2e} Msun, z = {z_val:.2f}")
plt.savefig('./tau_slice_xHI.pdf', bbox_inches='tight', dpi=300)
plt.show()

# test dependence on Mvir

plt.figure()
xHI_val = xHI_bins[5]
z_val = redshift_bins_wide[25]

cols = []
for i in range(len(halo_masses_Msun)):
    Mvir = halo_masses_Msun[i]
    col = cm.jet((Mvir - np.min(halo_masses_Msun))/ (np.max(halo_masses_Msun) - np.min(halo_masses_Msun)))
    cols.append(col)

for i in range(0,len(halo_masses_Msun)):
    tau_slice = tau_arr[:,5,i,25]  # select mid xHI and redshift
    plt.plot(coarse_freqs*HZ_TO_EV/1000, np.exp(-1*tau_slice), "-", color=cols[i], label=rf'Mvir = {halo_masses_Msun[i]:.2e} Msun') # plot in keV
plt.xlabel('Photon Energy (keV)')
plt.ylabel(r'Attenuation $e^{-\tau}$')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-4,1)
plt.xlim(1e-1,1e1)
norm = Normalize(vmin=np.min(halo_masses_Msun), vmax=np.max(halo_masses_Msun))
sm = cm.ScalarMappable(cmap=cm.jet, norm=norm)
sm.set_array(halo_masses_Msun)
ax = plt.gca()
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label(r'$M_\mathrm{vir}$ (Msun)')
plt.title(f"xHI = {xHI_val:.2e}, z = {z_val:.2f}")
plt.savefig('./tau_slice_Mvir.pdf', bbox_inches='tight', dpi=300)
plt.show()

# test dependence on z

plt.figure()
xHI_val = xHI_bins[5]
Mvir_val = halo_masses_Msun[25]

cols = []
for i in range(len(redshift_bins_wide)):
    z = redshift_bins_wide[i]
    col = cm.jet((z - np.min(redshift_bins_wide))/ (np.max(redshift_bins_wide) - np.min(redshift_bins_wide)))
    cols.append(col)

for i in range(0,len(redshift_bins_wide)):
    tau_slice = tau_arr[:,5,25,i]  # select mid xHI and halo mass
    plt.plot(coarse_freqs*HZ_TO_EV/1000, np.exp(-1*tau_slice), "-", color=cols[i]) # plot in keV
plt.xlabel('Photon Energy (keV)')
plt.ylabel(r'Attenuation $e^{-\tau}$')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-4,1)
plt.xlim(1e-1,1e1)
norm = Normalize(vmin=np.min(redshift_bins_wide), vmax=np.max(redshift_bins_wide))
sm = cm.ScalarMappable(cmap=cm.jet, norm=norm)
sm.set_array(redshift_bins_wide)
ax = plt.gca()
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label(r'$z$')
plt.title(f"Mvir = {Mvir_val:.2e} Msun, xHI = {xHI_val:.2e}")
plt.savefig('./tau_slice_z.pdf', bbox_inches='tight', dpi=300)
plt.show()

sio.savemat("halo_tau_grid.mat", {"tau_arr": tau_arr,
                                  "freq_bins": coarse_freqs,
                                  "redshift_bins": redshift_bins_wide,
                                  "halo_masses_Msun": halo_masses_Msun,
                                  "xHI_bins": xHI_bins})

'''
# average over xHI (dependence is weak)
tau_arr_avg_xHI = np.mean(tau_arr, axis=1)

sio.savemat("halo_tau_grid_avg_xHI.mat", {"tau_arr_avg_xHI": tau_arr_avg_xHI,
                                          "freq_bins": coarse_freqs,
                                          "redshift_bins": redshift_bins_wide,
                                          "halo_masses_Msun": halo_masses_Msun})
'''
# %%

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def fit_func(x0,x1,x2,x3): # vars are frequency, xHI, halo mass, redshift
    return (((x0 ** -0.20586985) + 0.28126824) * ((np.log(x2 ** 0.0954965) * np.log(x3)) + 196.60362)) + np.log(x0 ** -1.4258153)

print("reading data...")
read_data = sio.loadmat('halo_tau_grid.mat') # dimensions (n_freq, n_xHI, n_halo_masses, n_redshifts)
tau_grid = read_data['tau_arr'] # dimensions (n_freq, n_xHI, n_halo_masses, n_redshifts)
log_tau_grid = np.log10(tau_grid) # take log to handle wide range of values
coarse_freqs = read_data['freq_bins'].flatten() # dimensions (n_freq)
xHI_bins = read_data['xHI_bins'].flatten() # dimensions (n_xHI)
halo_masses_Msun = read_data['halo_masses_Msun'].flatten() # dimensions (n_halo_masses)
redshift_bins = read_data['redshift_bins'].flatten() # dimensions (n_redshifts)

# Plot the original tau values vs the fitted tau values at nu = 0.5 keV (or the closest frequency bin to that)
target_nu = 1 * 1e3 / (4.135667e-15)  # convert keV to Hz
nu_index = np.argmin(np.abs(coarse_freqs - target_nu))
logtau_values = log_tau_grid[nu_index,:,:,:].flatten()

# Create meshgrid for the specific frequency slice
# We use a single value for frequency, and full arrays for other dimensions
input_params = np.array(np.meshgrid([coarse_freqs[nu_index]], xHI_bins, halo_masses_Msun, redshift_bins, indexing='ij')).reshape(4, -1)
fitted_tau_values = fit_func(input_params[0], input_params[1], input_params[2], input_params[3])

print(logtau_values.shape) 
print(fitted_tau_values.shape)

tau_values = 10**logtau_values
fitted_tau_values_linear = 10**fitted_tau_values

plt.figure(figsize=(8,6))
#plt.scatter(logtau_values, fitted_tau_values, alpha=0.5)
#plt.plot([logtau_values.min(), logtau_values.max()], [logtau_values.min(), logtau_values.max()], 'r--')  # 1:1 line
plt.scatter(tau_values, fitted_tau_values_linear, alpha=0.5)
plt.plot([tau_values.min(), tau_values.max()], [tau_values.min(), tau_values.max()], 'r--')  # 1:1 line
#plt.xlabel('log10(Tau) from Grid')
#plt.ylabel('log10(Tau) from Fit')
plt.xlabel('Tau from Grid')
plt.ylabel('Tau from Fit')
plt.title('Fit vs Grid Tau Values at ~0.5 keV')
plt.savefig('fit_vs_grid_tau.pdf', bbox_inches='tight', dpi=300)
plt.show()

'''
# Plot original vs fitted tau values colorcoded by xHI
plt.figure(figsize=(8,6))
scatter = plt.scatter(logtau_values, fitted_tau_values, c=input_params[1], cmap='viridis', alpha=0.5)
plt.plot([logtau_values.min(), logtau_values.max()], [logtau_values.min(), logtau_values.max()], 'r--')  # 1:1 line
plt.xlabel('log10(Tau) from Grid')
plt.ylabel('log10(Tau) from Fit')
plt.title('Fit vs Grid Tau Values at ~0.5 keV, colored by xHI')
cbar = plt.colorbar(scatter)
cbar.set_label('xHI')
plt.savefig('fit_vs_grid_tau_xHI.pdf', bbox_inches='tight', dpi=300)
plt.show()
'''

'''
# Plot original vs fitted tau values colorcoded by halo mass
plt.figure(figsize=(8,6))
scatter = plt.scatter(logtau_values, fitted_tau_values, c=input_params[2], cmap='plasma', alpha=0.5)
plt.plot([logtau_values.min(), logtau_values.max()], [logtau_values.min(), logtau_values.max()], 'r--')  # 1:1 line
plt.xlabel('log10(Tau) from Grid')
plt.ylabel('log10(Tau) from Fit')
plt.title('Fit vs Grid Tau Values at ~0.5 keV, colored by Halo Mass')
cbar = plt.colorbar(scatter)
cbar.set_label('Halo Mass (Msun)')
plt.savefig('fit_vs_grid_tau_halomass.pdf', bbox_inches='tight', dpi=300)
plt.show()
'''

'''
# Plot original vs fitted tau values colorcoded by redshift
plt.figure(figsize=(8,6))
scatter = plt.scatter(logtau_values, fitted_tau_values, c=input_params[3], cmap='inferno', alpha=0.5)
plt.plot([logtau_values.min(), logtau_values.max()], [logtau_values.min(), logtau_values.max()], 'r--')  # 1:1 line
plt.xlabel('log10(Tau) from Grid')
plt.ylabel('log10(Tau) from Fit')
plt.title('Fit vs Grid Tau Values at ~0.5 keV, colored by Redshift')
cbar = plt.colorbar(scatter)
cbar.set_label('Redshift')
plt.savefig('fit_vs_grid_tau_redshift.pdf', bbox_inches='tight', dpi=300)
plt.show()
'''



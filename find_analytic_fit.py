import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.optimize import curve_fit

def product_of_PLs(input_params, A, gamma, delta):
    Mvir, z = input_params
    return A * (np.log10(Mvir)**gamma) * ((1+z)**delta)

read_data = sio.loadmat('./halo_tau_grid.mat')
taus = read_data['tau_arr']  # shape (num_freqs, num_xHI, num_Mvir, num_z)

# average over xHI axis
taus_avg_over_xHI = np.mean(taus, axis=1)  # shape (num_freqs, num_Mvir, num_z)

logtaus = np.log10(taus_avg_over_xHI)  # take log for fitting to handle wide range of values
coarse_freqs = read_data['freq_bins'].flatten()  # shape (num_freqs,)
xHI_bins = read_data['xHI_bins'].flatten()  # shape (num_xHI,)
halo_masses_Msun = read_data['halo_masses_Msun'].flatten()  # shape (num_Mvir,)
redshift_bins = read_data['redshift_bins'].flatten()  # shape (num_z,)

nu_idx = np.argmin(np.abs(coarse_freqs - 0.5*2.417990504024e17))  # index of frequency closest to 0.5 keV
logtaus_at_0p5keV = logtaus[nu_idx, :, :]  # shape (num_Mvir, num_z)

# Use curve_fit to fit the product of power laws to the tau grid
num_freqs, num_xHI, num_Mvir, num_z = taus.shape
input_params = np.array(np.meshgrid(halo_masses_Msun, redshift_bins, indexing='ij')).reshape(2, -1)
logtau_values = logtaus_at_0p5keV.flatten()
popt, pcov = curve_fit(product_of_PLs, input_params, logtau_values, p0=[-0.1, 1, 0.5], maxfev=100000000)
A_fit, gamma_fit, delta_fit = popt
print(f"Fitted parameters: A={A_fit:.3e}, gamma={gamma_fit:.3f}, delta={delta_fit:.3f}")

# Calculate the fitted tau values using the fitted parameters
fitted_tau_values = product_of_PLs(input_params, *popt)
fitted_tau_grid = fitted_tau_values.reshape(logtaus_at_0p5keV.shape)


# Plot the original tau values vs the fitted tau values
plt.figure(figsize=(8, 6))
plt.scatter(logtau_values, fitted_tau_values, alpha=0.5, s=10)
plt.plot([logtau_values.min(), logtau_values.max()], [logtau_values.min(), logtau_values.max()], 'r--')  # line y=x for reference
plt.xlabel('Original Log Tau Values')
plt.ylabel('Fitted Log Tau Values')
plt.title('Original vs Fitted Log Tau Values')
plt.xscale('linear')
plt.yscale('linear')
plt.grid(True)
plt.savefig('./tau_fit_comparison.pdf', bbox_inches='tight', dpi=300)
plt.show()

'''
# Plot the original tau values vs the fitted tau values with a colorbar for xHI
plt.figure(figsize=(8, 6))
scatter = plt.scatter(logtau_values, fitted_tau_values, c=input_params[0], cmap='viridis', alpha=0.5, s=10)
plt.plot([logtau_values.min(), logtau_values.max()], [logtau_values.min(), logtau_values.max()], 'r--')  # line y=x for reference
plt.xlabel('Original Log Tau Values')
plt.ylabel('Fitted Log Tau Values')
plt.title('Original vs Fitted Log Tau Values Colored by xHI')
plt.xscale('linear')
plt.yscale('linear')
plt.grid(True)
cbar = plt.colorbar(scatter)
cbar.set_label('xHI')
plt.savefig('./tau_fit_comparison_xHI.pdf', bbox_inches='tight', dpi=300)
plt.show()
'''

# Plot the original tau values vs the fitted tau values with a colorbar for Mvir
plt.figure(figsize=(8, 6))
scatter = plt.scatter(logtau_values, fitted_tau_values, c=input_params[0], cmap='plasma', alpha=0.5, s=10)
plt.plot([logtau_values.min(), logtau_values.max()], [logtau_values.min(), logtau_values.max()], 'r--')  # line y=x for reference
plt.xlabel('Original Log Tau Values')
plt.ylabel('Fitted Log Tau Values')
plt.title('Original vs Fitted Log Tau Values Colored by Mvir')
plt.xscale('linear')
plt.yscale('linear')
plt.grid(True)
cbar = plt.colorbar(scatter)
cbar.set_label('Mvir (Msun)')
plt.savefig('./tau_fit_comparison_Mvir.pdf', bbox_inches='tight', dpi=300)
plt.show()

# Plot the original tau values vs the fitted tau values with a colorbar for redshift
plt.figure(figsize=(8, 6))
scatter = plt.scatter(logtau_values, fitted_tau_values, c=input_params[1], cmap='inferno', alpha=0.5, s=10)
plt.plot([logtau_values.min(), logtau_values.max()], [logtau_values.min(), logtau_values.max()], 'r--')  # line y=x for reference
plt.xlabel('Original Log Tau Values')
plt.ylabel('Fitted Log Tau Values')
plt.title('Original vs Fitted Log Tau Values Colored by Redshift')
plt.xscale('linear')
plt.yscale('linear')
plt.grid(True)
cbar = plt.colorbar(scatter)
cbar.set_label('Redshift z')
plt.savefig('./tau_fit_comparison_redshift.pdf', bbox_inches='tight', dpi=300)
plt.show()
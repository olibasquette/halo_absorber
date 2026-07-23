import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pysr import PySRRegressor

read_data = sio.loadmat('halo_tau_grid.mat') # dimensions (n_freq, n_xHI, n_halo_masses, n_redshifts)
tau_grid = read_data['tau_arr'] # dimensions (n_freq, n_xHI, n_halo_masses, n_redshifts)
log_tau_grid = np.log10(tau_grid) # take log to handle wide range of values
coarse_freqs = read_data['freq_bins'].flatten() # dimensions (n_freq)
xHI_bins = read_data['xHI_bins'].flatten() # dimensions (n_xHI)
halo_masses_Msun = read_data['halo_masses_Msun'].flatten() # dimensions (n_halo_masses)
redshift_bins = read_data['redshift_bins'].flatten() # dimensions (n_redshifts)

model = PySRRegressor(
    niterations=1000,
    binary_operators=["+", "pow", "*", "/"],
    unary_operators=["exp", "log"],
    extra_sympy_mappings={"exp": "exp", "log": "log"},
    model_selection="best",
    loss="loss(x, y) = (x - y)^2",
    maxsize=20,
    batching=True,
    constraints={"pow": (-1, 1)},
    nested_constraints={"pow": {"pow": 0}}, # Don't nest powers
)

# Reshape the data for PySR
n_freq, n_xHI, n_halo_masses, n_redshifts = tau_grid.shape
X = np.zeros((n_freq * n_xHI * n_halo_masses * n_redshifts, 4))  # 4 features: freq, xHI, halo_mass, redshift
y = np.zeros(n_freq * n_xHI * n_halo_masses * n_redshifts)
index = 0
for i in range(n_freq):
    for j in range(n_xHI):
        for k in range(n_halo_masses):
            for l in range(n_redshifts):
                X[index] = [coarse_freqs[i], xHI_bins[j], halo_masses_Msun[k], redshift_bins[l]]
                y[index] = log_tau_grid[i, j, k, l]
                index += 1
                
model.fit(X, y)
print(model)


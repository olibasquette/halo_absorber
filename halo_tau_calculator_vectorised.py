#%% Script to calculate tau (halo optical depth) as a function of frequency, redshift, halo mass and ionisation fraction
# Eventually try using Zhu et al 2019 halo ionisation history to see how ionisation affects tau
# Also assume H and He are ionised to the same degree (Wyithe and Loeb 2003)

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from astropy import units as u
from astropy.cosmology import Planck15, z_at_value
import itertools
#from pysr import jl, PySRRegressor
#import juliapkg
#juliapkg.add("Zygote", uuid="e88e6eb3-aa80-5325-afca-941959d7151f")#
#juliapkg.resolve()

#%% Conversion factors and other constants

n_halo_masses = 50
n_xHIs = 10
MEGABARNS_TO_CM2 = 1e-18  # 1 Megabarn = 1e-18 cm^2
FH = 0.76  # Primordial fraction of baryonic mass in hydrogen
FHE = 0.24  # Primordial fraction of baryonic mass in helium
PROTON_MASS = 1.6726219e-24  # g
K_B = 1.380649e-16  # erg/K
G_CONST = 6.67430e-8  # cm^3 g^-1 s^-2
M_SUN_TO_GRAMS = 1.98847e33  # g
LITTLEH = 0.6774  # Planck 2015 value
HZ_TO_EV = 4.135667696e-15  # eV/Hz

#redshift_bins_wide = np.linspace(0.01,20,n_redshifts)
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
xHI_bins = np.concatenate(([0], np.logspace(-5,0,n_xHIs-2), [1]))  # neutral fraction in haloes

#%% Cross section fit parameters for H and He (add metals later)
HI_params = {
    'E0': 4.298e-1,
    'sig0': 5.475e4,
    'ya': 32.88,
    'P': 2.963,
    'yw': 0.0,
    'y0': 0.0,
    'y1': 0.0
} 

HeI_params = {
    'E0': 1.361e1,
    'sig0': 9.492e2,
    'ya': 1.469,
    'P': 3.188,
    'yw': 2.039,
    'y0': 0.4434,
    'y1': 2.136
}

HeII_params = {
    'E0': 1.720,
    'sig0': 1.369e4,
    'ya': 3.288e1,
    'P': 2.963,
    'yw': 0.0,
    'y0': 0.0,
    'y1': 0.0
}
#%% Get fits from Verner et al (1996) for photoionization cross-sections

def get_cross_section(
    E,  # Energy in eV
    E0, # eV
    sig0,  # Mb
    ya, # dimensionless
    P,  # dimensionless
    yw, # dimensionless
    y0, # dimensionless
    y1  # dimensionless
    ):
    x = E / E0 - y0
    y = np.sqrt(x**2 + y1**2)
    F = ((x-1)**2 + yw**2) * y**(0.5*P - 5.5) * (1 + np.sqrt(y/ya))**(-P)
    return sig0 * F  # Mb

#%% Formula for mu (mean molecular weight), accounting for ionisation of H and single ionisation of He (assume degree of ionisation of
# H and single ionisation of He are the same, Wyithe and Loeb 2003)
# Neglect double ionisation of He because it's annoying to calculate and doesn't affect mu very much

def get_mu(xHI):
    reciprocal_mu = FH * (xHI + 2*(1 - xHI)) + FHE * (xHI/4 + (1 - xHI)/2)
    return 1 / reciprocal_mu

#%% Formula for calculating gas virial temperature

def get_Tvir(mu, Mvir_Msun, rvir):
    Mvir = Mvir_Msun * M_SUN_TO_GRAMS  # g
    T = (1/2) * mu * PROTON_MASS * G_CONST * Mvir / (K_B * rvir)
    return T  # K

#%% Formula for calculating halo virial radius from mass and redshift

def get_r200(Mvir_Msun, rho_c):
    delta_c = 200  # Assumed overdensity for virial radius
    Mvir = Mvir_Msun * M_SUN_TO_GRAMS  # g
    rvir = (3 * Mvir / (4 * np.pi * delta_c * rho_c))**(1/3)  # cm
    return rvir

#%% Formula for gas density profile (assuming NFW and taking rvir = r200)

def f(x):
    return np.log(1 + x) - x /(1 + x)

def conc_fit_ragagnin(Mvir_Msun,z):
    val = 6.02 * (Mvir_Msun / 1e13)**(-0.12) * (1.47/(1 + z))**(0.16)
    return val

def conc_fit_duffy(x0,x1):
    pivot_mass = 2e12/LITTLEH   # Msun
    val = 5.71 * (x0/pivot_mass)**(-0.084) * (1+x1)**(-0.47)
    return val

def rho_gas(Mvir_Msun, r200, conc, rho_c, mu, Tvir, r):
    Mvir = Mvir_Msun * M_SUN_TO_GRAMS  # g
    c = conc
    delta_c = 200
    prefactor = (delta_c * rho_c)
    constants = -1*(G_CONST*mu*PROTON_MASS*Mvir)/(K_B*Tvir*f(c))
    #print(f"r200 shape: {r200.shape}, r shape: {r.shape}, prefactor shape: {prefactor.shape}, constants shape: {constants.shape}")
    r_bins = np.logspace(np.log10(r200), np.log10(r), 100)
    x_bins = r_bins / r200
    integrand = f(c * x_bins) / (r_bins**2)
    integral = np.trapz(integrand*r_bins, np.log(r_bins), axis=0)
    rho_gas_r = prefactor * np.exp(constants * integral)
    return rho_gas_r  # g/cm^3

#%% formula for the NFW dark matter density profile

def rho_NFW(z, Mvir_Msun, r):
    rho_c = Planck15.critical_density(z).to(u.g / u.cm**3).value  # g/cm^3
    delta_c = 200
    r200 = get_r200(Mvir_Msun, z)  # cm
    c = conc_fit_ragagnin(Mvir_Msun, z)
    rs = r200 / c # scale radius in cm
    return (rho_c * delta_c)/(r/rs * (1 + r/rs)**2)  # g/cm^3

# %%

# Adopt Ragagnin fit for the concentration parameter
# Calculate tau(z, Mvir, xHI, freq) using this fit

def get_integrand_grid(nu_arr, z_arr, Mvir_Msun_arr, xHI_arr):

    tau_arr = np.zeros((len(nu_arr),len(z_arr),len(Mvir_Msun_arr),len(xHI_arr)))
    
    # compute cross-sections for all nu
    cross_sections_HI = get_cross_section(nu_arr * HZ_TO_EV, *list(HI_params.values())) * MEGABARNS_TO_CM2  # cm^2
    cross_sections_HeI = get_cross_section(nu_arr * HZ_TO_EV, *list(HeI_params.values())) * MEGABARNS_TO_CM2  # cm^2
    cross_sections_HeII = get_cross_section(nu_arr * HZ_TO_EV, *list(HeII_params.values())) * MEGABARNS_TO_CM2  # cm^2
    #print(f"cross section array shape: {cross_sections_HI.shape}")
    #print(cross_sections_HI)

    # compute rho_c for all z
    rho_c_arr = Planck15.critical_density(z_arr).to(u.g / u.cm**3).value  # g/cm^3
    #print(f"rho_c array shape: {rho_c_arr.shape}")
    #print(rho_c_arr)

    # compute r200 for all Mvir and z
    r200_arr = get_r200(Mvir_Msun_arr[:,None], rho_c_arr[None,:])  # cm
    #print(f"r200 array shape: {r200_arr.shape}")
    #print(r200_arr)

    # compute concentration for all Mvir and z
    conc_arr = conc_fit_duffy(Mvir_Msun_arr[:,None], z_arr[None,:])  # dimensionless
    #print(f"concentration array shape: {conc_arr.shape}")
    #print(conc_arr)

    # compute grids of r values for integration (100 points from 0 to r200, logarithmically spaced) for each Mvir and z
    radii_arr = np.logspace(0,np.log10(r200_arr),100) # cm, shape (100, n_Mvir, n_z)
    radii_arr = np.moveaxis(radii_arr,0, -1)  # shape (n_Mvir, n_z, 100)
    #print(f"radii array shape: {radii_arr.shape}")
    #print(radii_arr)

    # compute mu for all xHI
    mu_arr = get_mu(xHI_arr)  # dimensionless
    #print(f"mu array shape: {mu_arr.shape}")
    #print(mu_arr)

    # compute Tvir for all xHI, Mvir, z
    Tvir_arr = get_Tvir(mu_arr[:,None,None], Mvir_Msun_arr[None,:,None], r200_arr[None,:,:])  # K
    #print(f"Tvir array shape: {Tvir_arr.shape}")
    #print(Tvir_arr)

    # compute rho_gas for all xHI, Mvir, z, r
    rho_gas_arr = rho_gas(Mvir_Msun_arr[None,:,None,None], r200_arr[None,:,:,None], conc_arr[None,:,:,None], rho_c_arr[None,None,:,None],
                          mu_arr[:,None,None,None], Tvir_arr[:,:,:,None], radii_arr[None,:,:,:])  # g/cm^3
    #print(f"rho_gas array shape: {rho_gas_arr.shape}")
    #print(rho_gas_arr)
    
    # compute integrand for all nu, xHI, Mvir, z, r
    integrand_arr = rho_gas_arr[None,:,:,:,:] * ((FH / PROTON_MASS) * xHI_arr[None,:,None,None,None] * cross_sections_HI[:,None,None,None,None] +
                                                 (FHE / (4*PROTON_MASS)) * xHI_arr[None,:,None,None,None]/4 * cross_sections_HeI[:,None,None,None,None] +
                                                 (FHE / (4*PROTON_MASS)) * (1 - xHI_arr[None,:,None,None,None])/2 * cross_sections_HeII[:,None,None,None,None])  # cm^-1
    #print(f"integrand array shape: {integrand_arr.shape}")
    #print(integrand_arr)
    return integrand_arr, radii_arr

#%%

integrand_arr, radii_arr = get_integrand_grid(coarse_freqs, redshift_bins_wide, halo_masses_Msun, xHI_bins)
np.save("integrand_arr.npy", integrand_arr)
np.save("radii_arr.npy", radii_arr)

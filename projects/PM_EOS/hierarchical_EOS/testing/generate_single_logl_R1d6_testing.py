import GWPhotonCounting
import jax.numpy as jnp
import jax
import numpy as np
import bilby
from bilby_cython.geometry import frequency_dependent_detector_tensor

import json
import sys
import os

from astropy.cosmology import Planck18
import astropy.units as u

# injected snr
idx = int(sys.argv[1])

from tqdm import tqdm

import jax
jax.config.update("jax_enable_x64", True)

frequencies = jnp.sort(jnp.fft.fftfreq(int(1e4), d=1/1e4))

# Setting up the two detectors to compare the 
detector_nosqz = GWPhotonCounting.detector.Detector(
    frequencies, '/home/ethan.payne/projects/GWPhotonCounting/examples/data/CE_shot_psd_nosqz.csv', 
    '/home/ethan.payne/projects/GWPhotonCounting/examples/data/CE_classical_psd.csv', 
    gamma=100, random_seed=1632, N_frequency_spaces=10, N_time_spaces=10)

detector_sqz = GWPhotonCounting.detector.Detector(
    frequencies, '/home/ethan.payne/projects/GWPhotonCounting/examples/data/CE_total_psd_sqz.csv', None, 
    gamma=100, random_seed=1632, N_frequency_spaces=10, N_time_spaces=10)

#Loading in the individual analysis from the sample
KNNModel = GWPhotonCounting.signal.PostMergerKNN(knn_file_path='/home/ethan.payne/code_libraries/apr4_knn_gw_model_2024/KNN_Models/APR4-knn_model-N100')
LorentzianModel = GWPhotonCounting.signal.PostMergerLorentzian()
dataset = np.genfromtxt(f'/home/ethan.payne/projects/GWPhotonCounting/projects/PM_EOS/hierarchical_EOS/bns_pm_dataset_MLE_250509.dat')

mtots, z, phi, psi, ra, dec, iota, f0_fit, gamma_fit, A_fit, phase_fit, snr, snr_sqz = dataset[idx]

amplitude_samples = dataset[:, 9] #* Planck18.luminosity_distance( dataset[:,1]).value /Planck18.luminosity_distance(z).value
gamma_samples = dataset[:, 8]

# Compute the expected number of photons and the strain
snr_desired = 0.5

PM_strain = KNNModel.generate_strain(detector_nosqz, frequencies, mtots, phi, z, ra, dec, iota, psi)
PM_scaling = snr_desired/detector_sqz.calculate_optimal_snr(PM_strain, frequencies)
PM_strain = PM_strain * PM_scaling

PM_strain_lorentzian = LorentzianModel.generate_strain(
    detector_nosqz, frequencies, f0=f0_fit, gamma=gamma_fit, A=A_fit, phase=phase_fit, t0=0)[0]
PM_scaling_lorentzian = snr_desired/detector_sqz.calculate_optimal_snr(PM_strain_lorentzian, frequencies)
PM_strain_lorentzian = PM_strain_lorentzian * PM_scaling_lorentzian


print('SNRs are: ', detector_sqz.calculate_optimal_snr(PM_strain, frequencies), detector_sqz.calculate_optimal_snr(PM_strain_lorentzian, frequencies))
print('Scaling factors are: ', PM_scaling, PM_scaling_lorentzian)


# # What I'm doing here is generating a Lorentzian signal with the same amplitude as the KNN model
# # I'm using this as a replacemenet/test because then the amplitude of the signal that we recover with and inject with is the same!  
# PM_strain = LorentzianModel.generate_strain(
#     detector_nosqz, frequencies, f0=f0_fit, gamma=gamma_fit, A=A_fit, phase=0, t0=0)[0]

# if detector_nosqz.calculate_optimal_snr(PM_strain, frequencies) > snr:
#     A = A_fit* snr / detector_nosqz.calculate_optimal_snr(PM_strain, frequencies)

#     PM_strain = LorentzianModel.generate_strain(
#         detector_nosqz, frequencies, f0=f0_fit, gamma=gamma_fit, A=A, phase=0, t0=0)[0]


# print(A_fit, , snr)

# Getting the frequencies
R1d6s = np.linspace(9,15,100) # 25
f0_R1d6s = GWPhotonCounting.hierarchical.frequency_model(mtots, R1d6s)/(1+z)

# Setting up the likelihood
poisson_likelihood = GWPhotonCounting.distributions.PoissonPhotonLikelihood()
noise_likelihood = GWPhotonCounting.distributions.PhaseQuadraturePhotonLikelihood() 
gaussian_likelihood = GWPhotonCounting.distributions.GaussianStrainLikelihood()
convolved_likelihood = GWPhotonCounting.distributions.MixturePhotonLikelihood(poisson_likelihood, noise_likelihood)

# Marginalizing over the likelihood
N_samples = 1000
N_t0s = 20


# Calculation for the CE1 detector
observed_strain = PM_strain + gaussian_likelihood.generate_realization(detector_sqz.total_psd, frequencies)
observed_strain_lorentzian = PM_strain_lorentzian + gaussian_likelihood.generate_realization(detector_sqz.total_psd, frequencies)


likelihood_event_i_strain = np.zeros(len(f0_R1d6s))
likelihood_event_i_strain_lorentzian = np.zeros(len(f0_R1d6s))
likelihood_event_i_strain_no_noise = np.zeros(len(f0_R1d6s))
likelihood_event_i_strain_no_noise_lorentzian = np.zeros(len(f0_R1d6s))

t0s = jnp.linspace(-0.02, 0.02, N_t0s)
sample_indexes = np.random.choice(10000, size=N_samples, replace=False)

phi0s = np.random.uniform(0, 2*np.pi, N_samples)

for l, f0 in enumerate(f0_R1d6s):
    likelihood_array_i_strain = np.zeros(N_samples)
    likelihood_array_i_strain_lorentzian = np.zeros(N_samples)
    likelihood_array_i_strain_no_noise = np.zeros(N_samples)
    likelihood_array_i_strain_no_noise_lorentzian = np.zeros(N_samples)

    print(l,f0)

    for j in range(N_samples):

        expected_strain = LorentzianModel.generate_strain(
            detector_nosqz, frequencies, f0=f0 + np.random.normal(loc=0, scale=61), gamma=gamma_samples[sample_indexes[j]], A=A_fit*PM_scaling,
            phase=phi0s[j], t0=t0s)

        expected_strain_lorentzian = LorentzianModel.generate_strain(
            detector_nosqz, frequencies, f0=f0 + np.random.normal(loc=0, scale=61), gamma=gamma_samples[sample_indexes[j]], A=A_fit*PM_scaling_lorentzian,
            phase=phi0s[j], t0=t0s)
        
    
        likelihood_array_i_strain[j] = gaussian_likelihood(observed_strain, expected_strain, detector_sqz.total_psd, frequencies)
        likelihood_array_i_strain_lorentzian[j] = gaussian_likelihood(observed_strain_lorentzian, expected_strain_lorentzian, detector_sqz.total_psd, frequencies)
        likelihood_array_i_strain_no_noise[j] = gaussian_likelihood(PM_strain, expected_strain, detector_sqz.total_psd, frequencies)
        likelihood_array_i_strain_no_noise_lorentzian[j] = gaussian_likelihood(PM_strain_lorentzian, expected_strain_lorentzian, detector_sqz.total_psd, frequencies)

    likelihood_event_i_strain[l] = jax.scipy.special.logsumexp(likelihood_array_i_strain) - jnp.log(len(likelihood_array_i_strain))
    likelihood_event_i_strain_lorentzian[l] = jax.scipy.special.logsumexp(likelihood_array_i_strain_lorentzian) - jnp.log(len(likelihood_array_i_strain_lorentzian))
    likelihood_event_i_strain_no_noise[l] = jax.scipy.special.logsumexp(likelihood_array_i_strain_no_noise) - jnp.log(len(likelihood_array_i_strain_no_noise))
    likelihood_event_i_strain_no_noise_lorentzian[l] = jax.scipy.special.logsumexp(likelihood_array_i_strain_no_noise_lorentzian) - jnp.log(len(likelihood_array_i_strain_no_noise_lorentzian))

data = {'logls_strain':list(likelihood_event_i_strain),
        'logls_strain_lorentzian':list(likelihood_event_i_strain_lorentzian),
        'logls_strain_no_noise':list(likelihood_event_i_strain_no_noise),
        'logls_strain_no_noise_lorentzian':list(likelihood_event_i_strain_no_noise_lorentzian),
        'mtot':mtots, 'z':z, 'phi':phi, 'psi':psi, 'ra':ra, 'dec':dec, 'iota':iota, 'f0_fit':f0_fit, 
        'gamma_fit':gamma_fit, 'A_fit':A_fit, 'phase_fit':phase_fit}

with open(f'results_250516a/result_CE_{idx}.json', 'w') as f:
    json.dump(data, f)

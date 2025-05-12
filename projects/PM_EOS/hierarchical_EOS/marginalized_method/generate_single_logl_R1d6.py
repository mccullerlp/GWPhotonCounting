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

frequencies = jnp.sort(jnp.fft.fftfreq(2**13, d=1/1e4))

# Setting up the two detectors to compare the 
detector_nosqz = GWPhotonCounting.detector.Detector(
    frequencies, '/home/ethan.payne/projects/GWPhotonCounting/examples/data/CE_shot_psd_nosqz.csv', 
    '/home/ethan.payne/projects/GWPhotonCounting/examples/data/CE_classical_psd.csv', 
    gamma=100, random_seed=1632, N_frequency_spaces=10)

detector_sqz = GWPhotonCounting.detector.Detector(
    frequencies, '/home/ethan.payne/projects/GWPhotonCounting/examples/data/CE_total_psd_sqz.csv', None, 
    gamma=100, random_seed=1632, N_frequency_spaces=10)

#Loading in the individual analysis from the sample
KNNModel = GWPhotonCounting.signal.PostMergerKNN(knn_file_path='/home/ethan.payne/code_libraries/apr4_knn_gw_model_2024/KNN_Models/APR4-knn_model-N100')
LorentzianModel = GWPhotonCounting.signal.PostMergerLorentzian()
dataset = np.genfromtxt(f'/home/ethan.payne/projects/GWPhotonCounting/projects/PM_EOS/hierarchical_EOS/bns_pm_dataset_MLE_250509.dat')

mtots, z, phi, psi, ra, dec, iota, f0_fit, gamma_fit, A_fit, phase_fit, snr, snr_sqz = dataset[idx]
print('SNRs are: ', snr, snr_sqz)

amplitude_samples = dataset[:, 9] #* Planck18.luminosity_distance( dataset[:,1]).value /Planck18.luminosity_distance(z).value
gamma_samples = dataset[:, 8]

# Compute the expected number of photons and the strain
PM_strain = KNNModel.generate_strain(detector_nosqz, frequencies, mtots, phi, z, ra, dec, iota, psi)

# # What I'm doing here is generating a Lorentzian signal with the same amplitude as the KNN model
# # I'm using this as a replacemenet/test because then the amplitude of the signal that we recover with and inject with is the same!  
# PM_strain = LorentzianModel.generate_strain(
#     detector_nosqz, frequencies, f0=f0_fit, gamma=gamma_fit, A=A_fit, phase=0, t0=0)[0]

# if detector_nosqz.calculate_optimal_snr(PM_strain, frequencies) > snr:
#     A = A_fit* snr / detector_nosqz.calculate_optimal_snr(PM_strain, frequencies)

#     PM_strain = LorentzianModel.generate_strain(
#         detector_nosqz, frequencies, f0=f0_fit, gamma=gamma_fit, A=A, phase=0, t0=0)[0]


# print(A_fit, detector_nosqz.calculate_optimal_snr(PM_strain, frequencies), snr)

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
observed_photons, signal_photons, noise_photons = convolved_likelihood.generate_realization(
        detector_nosqz.calculate_signal_photon_expectation(PM_strain, frequencies), 
        detector_nosqz.noise_photon_expectation)
_, _, noise_photons_0d01 = convolved_likelihood.generate_realization(
        detector_nosqz.calculate_signal_photon_expectation(PM_strain, frequencies),
        0.01 * detector_nosqz.noise_photon_expectation)
_, _, noise_photons_0d1 = convolved_likelihood.generate_realization(
        detector_nosqz.calculate_signal_photon_expectation(PM_strain, frequencies),
        0.1 * detector_nosqz.noise_photon_expectation)
_, _, noise_photons_0d5 = convolved_likelihood.generate_realization(
        detector_nosqz.calculate_signal_photon_expectation(PM_strain, frequencies),
        0.5 * detector_nosqz.noise_photon_expectation)
observed_photons_no_background = signal_photons
observed_strain = PM_strain + gaussian_likelihood.generate_realization(detector_sqz.total_psd, frequencies)
observed_strain_15db = PM_strain + gaussian_likelihood.generate_realization(10**(-0.5) * detector_sqz.total_psd, frequencies)
observed_strain_20db = PM_strain + gaussian_likelihood.generate_realization(10**(-1) * detector_sqz.total_psd, frequencies)

likelihood_event_i = np.zeros(len(f0_R1d6s))
likelihood_event_i_0d01 = np.zeros(len(f0_R1d6s))
likelihood_event_i_0d1 = np.zeros(len(f0_R1d6s))
likelihood_event_i_0d5 = np.zeros(len(f0_R1d6s))
likelihood_event_i_no_background = np.zeros(len(f0_R1d6s))
likelihood_event_i_strain = np.zeros(len(f0_R1d6s))
likelihood_event_i_strain_15db = np.zeros(len(f0_R1d6s))
likelihood_event_i_strain_20db = np.zeros(len(f0_R1d6s))

t0s = jnp.linspace(-0.02, 0.02, N_t0s)
sample_indexes = np.random.choice(10000, size=N_samples, replace=False)

phi0s = np.random.uniform(0, 2*np.pi, N_samples)

for l, f0 in enumerate(f0_R1d6s):

    likelihood_array_i = np.zeros(N_samples)
    likelihood_array_i_0d01 = np.zeros(N_samples)
    likelihood_array_i_0d1 = np.zeros(N_samples)
    likelihood_array_i_0d5 = np.zeros(N_samples)
    likelihood_array_i_no_background = np.zeros(N_samples)
    likelihood_array_i_strain = np.zeros(N_samples)
    likelihood_array_i_strain_15db = np.zeros(N_samples)
    likelihood_array_i_strain_20db = np.zeros(N_samples)

    print(l,f0)

    for j in range(N_samples):

        expected_photon_count_signal = LorentzianModel.generate_photon_count(
            detector_nosqz, frequencies, f0=f0, gamma=gamma_samples[sample_indexes[j]], A=amplitude_samples[sample_indexes[j]],
            phase=phi0s[j], t0=t0s)
        expected_strain = LorentzianModel.generate_strain(
            detector_nosqz, frequencies, f0=f0, gamma=gamma_samples[sample_indexes[j]], A=amplitude_samples[sample_indexes[j]],
            phase=phi0s[j], t0=t0s)
        
        
        likelihood_array_i[j] = convolved_likelihood(
            signal_photons+noise_photons, expected_photon_count_signal, detector_nosqz.noise_photon_expectation)
        likelihood_array_i_0d01[j] = convolved_likelihood(
            signal_photons+noise_photons_0d01, expected_photon_count_signal, 0.01 * detector_nosqz.noise_photon_expectation)
        likelihood_array_i_0d1[j] = convolved_likelihood(
            signal_photons+noise_photons_0d1, expected_photon_count_signal, 0.1 * detector_nosqz.noise_photon_expectation)
        likelihood_array_i_0d5[j] = convolved_likelihood(
            signal_photons+noise_photons_0d5, expected_photon_count_signal, 0.5 * detector_nosqz.noise_photon_expectation)
        likelihood_array_i_no_background[j] = poisson_likelihood(
            observed_photons_no_background, expected_photon_count_signal)
    
        likelihood_array_i_strain[j] = gaussian_likelihood(observed_strain, expected_strain, detector_sqz.total_psd, frequencies)
        likelihood_array_i_strain_15db[j] = gaussian_likelihood(observed_strain_15db, expected_strain, 10**(-0.5) * detector_sqz.total_psd, frequencies)
        likelihood_array_i_strain_20db[j] = gaussian_likelihood(observed_strain_20db, expected_strain, 10**(-1) * detector_sqz.total_psd, frequencies)

    likelihood_event_i[l] = jax.scipy.special.logsumexp(likelihood_array_i) - jnp.log(len(likelihood_array_i))
    likelihood_event_i_0d01[l] = jax.scipy.special.logsumexp(likelihood_array_i_0d01) - jnp.log(len(likelihood_array_i_0d01))
    likelihood_event_i_0d1[l] = jax.scipy.special.logsumexp(likelihood_array_i_0d1) - jnp.log(len(likelihood_array_i_0d1))
    likelihood_event_i_0d5[l] = jax.scipy.special.logsumexp(likelihood_array_i_0d5) - jnp.log(len(likelihood_array_i_0d5))
    likelihood_event_i_no_background[l] = jax.scipy.special.logsumexp(likelihood_array_i_no_background) - jnp.log(len(likelihood_array_i_no_background))
    likelihood_event_i_strain[l] = jax.scipy.special.logsumexp(likelihood_array_i_strain) - jnp.log(len(likelihood_array_i_strain))
    likelihood_event_i_strain_15db[l] = jax.scipy.special.logsumexp(likelihood_array_i_strain_15db) - jnp.log(len(likelihood_array_i_strain_15db))
    likelihood_event_i_strain_20db[l] = jax.scipy.special.logsumexp(likelihood_array_i_strain_20db) - jnp.log(len(likelihood_array_i_strain_20db))

data = {'logls':list(likelihood_event_i), 
        'logls_0d01':list(likelihood_event_i_0d01),
        'logls_0d1':list(likelihood_event_i_0d1),
        'logls_0d5':list(likelihood_event_i_0d5),
        'logls_no_background':list(likelihood_event_i_no_background), 
        'logls_strain':list(likelihood_event_i_strain), 
        'logls_strain_15db':list(likelihood_event_i_strain_15db),
        'logls_strain_20db':list(likelihood_event_i_strain_20db),
        'n_signal_photons':float(jnp.sum(signal_photons)),
        'n_noise_photons':float(jnp.sum(noise_photons)),
        'n_noise_photons_0d01':float(jnp.sum(noise_photons_0d01)),
        'n_noise_photons_0d1':float(jnp.sum(noise_photons_0d1)),
        'n_noise_photons_0d5':float(jnp.sum(noise_photons_0d5)),
        'snr':float(snr), 'snr_sqz':float(snr_sqz), 'mtot':mtots, 'z':z, 'phi':phi, 'psi':psi, 'ra':ra, 'dec':dec, 'iota':iota, 'f0_fit':f0_fit, 
        'gamma_fit':gamma_fit, 'A_fit':A_fit, 'phase_fit':phase_fit}

with open(f'results_250512/result_CE_{idx}.json', 'w') as f:
    json.dump(data, f)

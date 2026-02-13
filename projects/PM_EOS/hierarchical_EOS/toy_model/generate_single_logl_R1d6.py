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
LorentzianModel = GWPhotonCounting.signal.PostMergerLorentzian()
dataset = np.genfromtxt(f'/home/ethan.payne/projects/GWPhotonCounting/projects/PM_EOS/hierarchical_EOS/bns_pm_dataset_MLE_250509.dat')

mtots, z, phi, psi, ra, dec, iota, f0_fit, gamma_fit, A_fit, phase_fit, snr, snr_sqz = dataset.T

mtot_i = np.random.choice(mtots, size=1)[0]
z_i = np.random.choice(z, size=1)[0]
phi_i = np.random.uniform(0, 2*np.pi, 1)[0]
epsilon_i = np.random.normal(loc=0, scale=61, size=1)[0]
gamma_i = np.random.choice(gamma_fit, size=1)[0]
A_i = np.random.choice(A_fit, size=1)[0]
t0_i = np.random.uniform(-0.02, 0.02, 1)[0]

f0_i = GWPhotonCounting.hierarchical.frequency_model(mtot_i, 11.27)/(1+z_i) + epsilon_i

PM_strain = LorentzianModel.generate_strain(detector_nosqz, frequencies, f0=f0_i, gamma=gamma_i, A=A_i, phase=phi_i, t0=t0_i)[0]

snr = detector_nosqz.calculate_optimal_snr(PM_strain, frequencies)
snr_sqz = detector_sqz.calculate_optimal_snr(PM_strain, frequencies)
print('SNRs are: ', snr, snr_sqz)

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
f0_R1d6s = GWPhotonCounting.hierarchical.frequency_model(mtot_i, R1d6s)/(1+z_i)

# Setting up the likelihood
poisson_likelihood = GWPhotonCounting.distributions.PoissonPhotonLikelihood()
noise_likelihood = GWPhotonCounting.distributions.PhaseQuadraturePhotonLikelihood() 
gaussian_likelihood = GWPhotonCounting.distributions.GaussianStrainLikelihood()
convolved_likelihood = GWPhotonCounting.distributions.MixturePhotonLikelihood(poisson_likelihood, noise_likelihood)

# Marginalizing over the likelihood
N_samples = 1000
N_t0s = 20

n_exp = np.sum(detector_nosqz.calculate_signal_photon_expectation(PM_strain, frequencies))
print('Expected number of photons: ', n_exp, detector_nosqz.calculate_optimal_snr(PM_strain, frequencies, fmin=1.5e3)**2/2)
print('Ratio: ', n_exp/(detector_nosqz.calculate_optimal_snr(PM_strain, frequencies, fmin=1.5e3)**2/2), f0_i)
# Calculation for the CE1 detector
observed_photons, signal_photons, noise_photons = convolved_likelihood.generate_realization(
        detector_nosqz.calculate_signal_photon_expectation(PM_strain, frequencies), 
        detector_nosqz.noise_photon_expectation)
_, _, noise_photons_0d1 = convolved_likelihood.generate_realization(
        detector_nosqz.calculate_signal_photon_expectation(PM_strain, frequencies),
        0.1 * detector_nosqz.noise_photon_expectation)
_, _, noise_photons_0d3 = convolved_likelihood.generate_realization(
        detector_nosqz.calculate_signal_photon_expectation(PM_strain, frequencies),
        0.3 * detector_nosqz.noise_photon_expectation)
observed_photons_no_background = signal_photons
observed_strain = PM_strain + gaussian_likelihood.generate_realization(detector_sqz.total_psd, frequencies)
observed_strain_15db = PM_strain + gaussian_likelihood.generate_realization(10**(-0.5) * detector_sqz.total_psd, frequencies)

likelihood_event_i = np.zeros(len(f0_R1d6s))
likelihood_event_i_0d1 = np.zeros(len(f0_R1d6s))
likelihood_event_i_0d3 = np.zeros(len(f0_R1d6s))
likelihood_event_i_strain = np.zeros(len(f0_R1d6s))
likelihood_event_i_strain_15db = np.zeros(len(f0_R1d6s))

likelihood_event_i_margA = np.zeros(len(f0_R1d6s))
likelihood_event_i_margA_0d1 = np.zeros(len(f0_R1d6s))
likelihood_event_i_margA_0d3 = np.zeros(len(f0_R1d6s))
likelihood_event_i_margA_strain = np.zeros(len(f0_R1d6s))
likelihood_event_i_margA_strain_15db = np.zeros(len(f0_R1d6s))

t0s = jnp.linspace(-0.02, 0.02, N_t0s)

phi0s = np.random.uniform(0, 2*np.pi, N_samples)
epsilons = np.random.normal(loc=0, scale=61, size=N_samples)
gamma_samples = np.random.choice(gamma_fit, size=N_samples)
amplitude_samples = np.random.choice(A_fit, size=N_samples)

for l, f0 in enumerate(f0_R1d6s):

    likelihood_array_i = np.zeros(N_samples)
    likelihood_array_i_0d1 = np.zeros(N_samples)
    likelihood_array_i_0d3 = np.zeros(N_samples)
    likelihood_array_i_strain = np.zeros(N_samples)
    likelihood_array_i_strain_15db = np.zeros(N_samples)

    likelihood_array_i_margA = np.zeros(N_samples)
    likelihood_array_i_margA_0d1 = np.zeros(N_samples)
    likelihood_array_i_margA_0d3 = np.zeros(N_samples)
    likelihood_array_i_margA_strain = np.zeros(N_samples)
    likelihood_array_i_margA_strain_15db = np.zeros(N_samples)

    print(l,f0)

    for j in range(N_samples):

        expected_photon_count_signal = LorentzianModel.generate_photon_count(
            detector_nosqz, frequencies, f0=f0 + epsilons[j], gamma=gamma_samples[j], A=A_i,
            phase=phi0s[j], t0=t0s)
        expected_strain = LorentzianModel.generate_strain(
            detector_nosqz, frequencies, f0=f0 + epsilons[j], gamma=gamma_samples[j], A=A_i,
            phase=phi0s[j], t0=t0s)
        
        expected_photon_count_signal_margA = LorentzianModel.generate_photon_count(
            detector_nosqz, frequencies, f0=f0 + epsilons[j], gamma=gamma_samples[j], A=amplitude_samples[j],
            phase=phi0s[j], t0=t0s)
        expected_strain_margA = LorentzianModel.generate_strain(
            detector_nosqz, frequencies, f0=f0 + epsilons[j], gamma=gamma_samples[j], A=amplitude_samples[j],
            phase=phi0s[j], t0=t0s)
        
        likelihood_array_i[j] = convolved_likelihood(
            signal_photons+noise_photons, expected_photon_count_signal, detector_nosqz.noise_photon_expectation)
        likelihood_array_i_0d1[j] = convolved_likelihood(
            signal_photons+noise_photons_0d1, expected_photon_count_signal, 0.1 * detector_nosqz.noise_photon_expectation)
        likelihood_array_i_0d3[j] = convolved_likelihood(
            signal_photons+noise_photons_0d3, expected_photon_count_signal, 0.5 * detector_nosqz.noise_photon_expectation)
    
        likelihood_array_i_strain[j] = gaussian_likelihood(observed_strain, expected_strain, detector_sqz.total_psd, frequencies)
        likelihood_array_i_strain_15db[j] = gaussian_likelihood(observed_strain_15db, expected_strain, 10**(-0.5) * detector_sqz.total_psd, frequencies)


        likelihood_array_i_margA[j] = convolved_likelihood(
            signal_photons+noise_photons, expected_photon_count_signal_margA, detector_nosqz.noise_photon_expectation)
        likelihood_array_i_margA_0d1[j] = convolved_likelihood(
            signal_photons+noise_photons_0d1, expected_photon_count_signal_margA, 0.1 * detector_nosqz.noise_photon_expectation)
        likelihood_array_i_margA_0d3[j] = convolved_likelihood(
            signal_photons+noise_photons_0d3, expected_photon_count_signal_margA, 0.5 * detector_nosqz.noise_photon_expectation)
        
        likelihood_array_i_margA_strain[j] = gaussian_likelihood(observed_strain, expected_strain_margA, detector_sqz.total_psd, frequencies)
        likelihood_array_i_margA_strain_15db[j] = gaussian_likelihood(observed_strain_15db, expected_strain_margA, 10**(-0.5) * detector_sqz.total_psd, frequencies)

    likelihood_event_i[l] = jax.scipy.special.logsumexp(likelihood_array_i) - jnp.log(len(likelihood_array_i))
    likelihood_event_i_0d1[l] = jax.scipy.special.logsumexp(likelihood_array_i_0d1) - jnp.log(len(likelihood_array_i_0d1))
    likelihood_event_i_0d3[l] = jax.scipy.special.logsumexp(likelihood_array_i_0d3) - jnp.log(len(likelihood_array_i_0d3))

    likelihood_event_i_strain[l] = jax.scipy.special.logsumexp(likelihood_array_i_strain) - jnp.log(len(likelihood_array_i_strain))
    likelihood_event_i_strain_15db[l] = jax.scipy.special.logsumexp(likelihood_array_i_strain_15db) - jnp.log(len(likelihood_array_i_strain_15db))

    likelihood_event_i_margA[l] = jax.scipy.special.logsumexp(likelihood_array_i_margA) - jnp.log(len(likelihood_array_i_margA))
    likelihood_event_i_margA_0d1[l] = jax.scipy.special.logsumexp(likelihood_array_i_margA_0d1) - jnp.log(len(likelihood_array_i_margA_0d1))
    likelihood_event_i_margA_0d3[l] = jax.scipy.special.logsumexp(likelihood_array_i_margA_0d3) - jnp.log(len(likelihood_array_i_margA_0d3))

    likelihood_event_i_margA_strain[l] = jax.scipy.special.logsumexp(likelihood_array_i_margA_strain) - jnp.log(len(likelihood_array_i_margA_strain))
    likelihood_event_i_margA_strain_15db[l] = jax.scipy.special.logsumexp(likelihood_array_i_margA_strain_15db) - jnp.log(len(likelihood_array_i_margA_strain_15db))


data = {'logls':list(likelihood_event_i), 
        'logls_0d1':list(likelihood_event_i_0d1),
        'logls_0d3':list(likelihood_event_i_0d3),
        'logls_strain':list(likelihood_event_i_strain), 
        'logls_strain_15db':list(likelihood_event_i_strain_15db),
        'logls_margA':list(likelihood_event_i_margA),
        'logls_margA_0d1':list(likelihood_event_i_margA_0d1),
        'logls_margA_0d3':list(likelihood_event_i_margA_0d3),
        'logls_margA_strain':list(likelihood_event_i_margA_strain),
        'logls_margA_strain_15db':list(likelihood_event_i_margA_strain_15db),
        'n_signal_photons':float(jnp.sum(signal_photons)),
        'n_noise_photons':float(jnp.sum(noise_photons)),
        'n_noise_photons_0d1':float(jnp.sum(noise_photons_0d1)),
        'n_noise_photons_0d3':float(jnp.sum(noise_photons_0d3)),
        'snr':float(snr), 'snr_sqz':float(snr_sqz), 'n_exp':float(n_exp),
        'mtot':mtot_i, 'z':z_i, 'phi':phi_i, 'epsilon' 'f0':f0_i, 
        'gamma_fit':gamma_i, 'A_fit':A_i, 'phase_fit':phi_i}

with open(f'results_250520/result_CE_{idx}.json', 'w') as f:
    json.dump(data, f)

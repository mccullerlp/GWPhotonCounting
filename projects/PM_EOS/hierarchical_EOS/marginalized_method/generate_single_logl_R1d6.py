import GWPhotonCounting
import jax.numpy as jnp
import jax
import numpy as np
import bilby
from bilby_cython.geometry import frequency_dependent_detector_tensor

import json
import sys
import os

from scipy.optimize import minimize

from astropy.cosmology import Planck18
import astropy.units as u

# injected snr
idx = int(sys.argv[1])

from tqdm import tqdm

import jax

import os
directory_path = "results_260215b/"
os.makedirs(directory_path, exist_ok=True)

fname = directory_path + f'result_CE_{idx}.json'
if os.path.exists(fname):
    print(f"Output file {fname} exists")
    sys.exit(0)


jax.config.update("jax_enable_x64", True)

frequencies = jnp.sort(jnp.fft.fftfreq(2**12, d=1/1e4))

# TODO make this work on non-linear frequencies array
dF = frequencies[1] - frequencies[0]
duration = 1/dF
print("DURATION: ", duration)

#frequencies = jnp.sort(jnp.fft.fftfreq(2**12, d=1/1e4))
print("freq max", np.max(frequencies))
print("dF", frequencies[-1] - frequencies[-2])

# Setting up the two detectors to compare the
detector_nosqz = GWPhotonCounting.detector.Detector(
    frequencies, '/home/mcculler/projects/GWPhotonCounting/examples/data/CE_shot_psd_nosqz.csv',
    '/home/mcculler/projects/GWPhotonCounting/examples/data/CE_classical_psd.csv',
    gamma=80, random_seed=1632, N_frequency_spaces=17*2, N_time_spaces=8)

detector_sqz = GWPhotonCounting.detector.Detector(
    frequencies, '/home/mcculler/projects/GWPhotonCounting/examples/data/CE_total_psd_sqz.csv', None,
    gamma=80, random_seed=1632, N_frequency_spaces=17*2, N_time_spaces=8)


#Loading in the individual analysis from the sample
KNNModel = GWPhotonCounting.signal.PostMergerKNN(knn_file_path='/home/mcculler/photon_counting/apr4_knn_gw_model_2024/KNN_Models/APR4-knn_model-N100')
LorentzianModel = GWPhotonCounting.signal.PostMergerLorentzian()
dataset = np.genfromtxt(f'/home/mcculler/projects/GWPhotonCounting/projects/PM_EOS/hierarchical_EOS/bns_pm_dataset_MLE_260213.dat') #bns_pm_dataset_MLE_250609
#dataset = np.genfromtxt(f'/home/mcculler/projects/GWPhotonCounting/projects/PM_EOS/hierarchical_EOS/bns_pm_dataset_MLE_250609.dat') #bns_pm_dataset_MLE_250609

# OK, now sorting on SNR for the first 10000!!!
if idx <= 10000:
    mtots, z, phi, psi, ra, dec, iota, f0_fit, gamma_fit, A_fit, phase_fit, snr, snr_sqz = dataset[:10000].T
    #print(dataset.shape)
    #print(snr.shape)
    snr_idx = np.argsort(-snr)
    #print(snr_idx.shape)
    sorted_dataset = dataset[snr_idx]
    #print(sorted_dataset.shape)
    mtots, z, phi, psi, ra, dec, iota, f0_fit, gamma_fit, A_fit, phase_fit, snr, snr_sqz = sorted_dataset[idx]
else:
    mtots, z, phi, psi, ra, dec, iota, f0_fit, gamma_fit, A_fit, phase_fit, snr, snr_sqz = dataset[idx]

amplitude_samples = dataset[:, 9] #* Planck18.luminosity_distance( dataset[:,1]).value /Planck18.luminosity_distance(z).value
gamma_samples = dataset[:, 8]

# Compute the expected number of photons and the strain
PM_strain = KNNModel.generate_strain(detector_nosqz, frequencies, mtots, phi, z, ra, dec, iota, psi)

snr = detector_nosqz.calculate_optimal_snr(PM_strain, frequencies)
snr_sqz = detector_sqz.calculate_optimal_snr(PM_strain, frequencies)
print('SNRs are: ', snr, snr_sqz)
print('f0_fit: ', f0_fit)

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
R1d6s = np.linspace(10,14,100) # 25
f0_R1d6s = GWPhotonCounting.hierarchical.frequency_model(mtots, R1d6s)/(1+z)

# Setting up the likelihood
poisson_likelihood = GWPhotonCounting.distributions.PoissonPhotonLikelihood()
noise_likelihood = GWPhotonCounting.distributions.PhaseQuadraturePhotonLikelihood()
gaussian_likelihood = GWPhotonCounting.distributions.GaussianStrainLikelihood()
convolved_likelihood = GWPhotonCounting.distributions.MixturePhotonLikelihood(poisson_likelihood, noise_likelihood)

# Marginalizing over the likelihood
N_samples = 100
N_t0s = 10

np.random.seed()
n_exp = np.sum(detector_nosqz.calculate_signal_photon_expectation(PM_strain, frequencies))
n_self = detector_nosqz.calculate_signal_photon_expectation_self(PM_strain, frequencies)
print('Expected number of photons (vs HD), (into templates), (full power): ', snr**2/4, n_exp, n_self)
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

likelihood_event_i = np.zeros(len(f0_R1d6s))
likelihood_event_i_0d01 = np.zeros(len(f0_R1d6s))
likelihood_event_i_0d1 = np.zeros(len(f0_R1d6s))
likelihood_event_i_0d5 = np.zeros(len(f0_R1d6s))
likelihood_event_i_no_background = np.zeros(len(f0_R1d6s))
likelihood_event_i_strain = np.zeros(len(f0_R1d6s))
likelihood_event_i_strain_15db = np.zeros(len(f0_R1d6s))
likelihood_event_i_strain_20db = np.zeros(len(f0_R1d6s))
likelihood_event_i_strainMx = np.zeros(len(f0_R1d6s))
likelihood_event_i_strain_15dbMx = np.zeros(len(f0_R1d6s))
likelihood_event_i_strain_20dbMx = np.zeros(len(f0_R1d6s))

t0s = jnp.linspace(-0.01, 0.03, N_t0s)

# Attempt at fixing the calculation in the hierarchical analysis.....
def neg_logl_cost_function(x0, strain, frequencies, noise_psd):

    f0, gamma, log10A, phase, t0 = x0

    t0s = np.array([t0])

    # Calculate the expected strain
    lorentzian_strain = LorentzianModel.generate_strain(
        detector_sqz, frequencies, f0=f0, gamma=gamma, A=10**log10A, phase=phase, t0=t0s)

    # Calculate the log-likelihood
    logl = gaussian_likelihood(strain, lorentzian_strain, noise_psd,frequencies)

    # Return the negative logl
    return -logl


def neg_logl_cost_function_split(x0, strain, frequencies):

    f0, gamma, log10A, phase = x0

    t0s = jnp.linspace(-0.02, 0.02, 100)

    # Calculate the expected strain
    lorentzian_strain = LorentzianModel.generate_strain(
        detector_nosqz, frequencies, f0=f0, gamma=gamma, A=10**log10A, phase=phase, t0=t0s)

    # Return the negative logl
    t0s, logl = np.array([
		[t0s[i], gaussian_likelihood(strain, lorentzian_strain[i].reshape(1, -1), jnp.ones(len(frequencies))*1e-55,frequencies)]
		for i in range(len(t0s))
	]).T
    midx = np.argmin(-logl)
    return t0s[midx], logl[midx]


result_fits = []
neg_logls = []
for i in tqdm(range(100)):
    params_0 = [np.abs(np.random.uniform(.995,1.005) * frequencies[np.argmax(np.abs(PM_strain))]), np.random.uniform(1,100), np.random.uniform(-10,1), np.random.uniform(0,2*np.pi), np.random.uniform(-.02, .05)]
    minimize_result = minimize(neg_logl_cost_function,
                                x0=params_0,
                                args=(PM_strain/np.max(np.abs(PM_strain)), frequencies, np.ones_like(frequencies)),
                                bounds=((500, 5000), (0, 200), (-5, 5), (0, 2*np.pi), (-.05, .1)))

    result_fits.append(minimize_result.x)
    neg_logls.append(minimize_result.fun)

fpeak_fit, gamma_fit, log10A_fit_pc, phase_fit, t0_fit = result_fits[np.argmin(neg_logls)]
print('fpeak_fit: ', fpeak_fit)
print('amp_fit: ', 10**log10A_fit_pc)
print("T0:", t0_fit)
#which = neg_logl_cost_function_split((fpeak_fit, gamma_fit, log10A_fit_pc, phase_fit), PM_strain, frequencies)
#t0_fit, logl = which

neg_logl_pc = np.min(neg_logls)
fitted_signal = LorentzianModel.generate_strain(
        detector_sqz, frequencies, f0=fpeak_fit, gamma=gamma_fit, A=10**log10A_fit_pc * np.max(np.abs(PM_strain)), phase=phase_fit, t0=t0_fit)[0]

snr_fit_pc = detector_nosqz.calculate_optimal_snr(fitted_signal, frequencies)
snr_cross = detector_nosqz.calculate_inner_product(PM_strain, fitted_signal, frequencies)
SNR_frac = (snr_cross**2 / snr_fit_pc / snr)
print("SNR, SNR_L, SNR_X, fraction_overlap: ", snr, snr_fit_pc, snr_cross, SNR_frac)

# result_fits = []
# neg_logls = []
# #params_0 = [np.abs(frequencies[np.argmax(np.abs(PM_strain))]), np.random.uniform(0,200), jnp.log10(np.max(np.abs(PM_strain))), np.random.uniform(0,2*np.pi)]
# for i in tqdm(range(100)):
#     params_0 = [np.random.uniform(500,5000), np.random.uniform(0,200), np.log10(A_fit), np.random.uniform(0,2*np.pi)]
#     minimize_result = minimize(neg_logl_cost_function,
#                                 x0=params_0,
#                                 args=(PM_strain, frequencies, detector_sqz.total_psd),
#                                 bounds=((500, 5000), (0, 200), (-40, -21), (0, 2*np.pi)))
#     _, _, log10A_fit, _ = minimize_result.x

#     result_fits.append(minimize_result.x)
#     neg_logls.append(minimize_result.fun)

# fpeak_fit, gamma_fit, log10A_fit_strain, phase_fit = result_fits[np.argmin(neg_logls)]
# neg_logl_strain = np.min(neg_logls)
# fitted_signal = LorentzianModel.generate_strain(
#         detector_sqz, frequencies, f0=fpeak_fit, gamma=gamma_fit, A=10**log10A_fit_strain, phase=phase_fit, t0=0)[0]
# snr_fit_strain = detector_sqz.calculate_optimal_snr(fitted_signal, frequencies)

# log10A_fits = []
# neg_logls = []
# for i in tqdm(range(100)):
#     params_0 = [np.random.uniform(500,5000), np.random.uniform(0,200), np.log10(A_fit), np.random.uniform(0,2*np.pi)]
#     minimize_result = minimize(neg_logl_cost_function,
#                                 x0=params_0,
#                                 args=(PM_strain, frequencies, 10**(-0.5) * detector_sqz.total_psd),
#                                 bounds=((500, 5000), (0, 200), (-40, -21), (0, 2*np.pi)))
#     _, _, log10A_fit, _ = minimize_result.x

#     result_fits.append(minimize_result.x)
# #     neg_logls.append(minimize_result.fun)

# fpeak_fit, gamma_fit, log10A_fit_strain_15db, phase_fit = result_fits[np.argmin(neg_logls)]
# neg_logl_strain_15db = np.min(neg_logls)
# fitted_signal = LorentzianModel.generate_strain(
#         detector_sqz, frequencies, f0=fpeak_fit, gamma=gamma_fit, A=10**log10A_fit_strain_15db, phase=phase_fit, t0=0)[0]
# snr_fit_strain_15db = detector_sqz.calculate_optimal_snr(fitted_signal, frequencies) *(10**(-0.5))

print('Log10A fit is: ', A_fit, 10**log10A_fit_pc * np.max(np.abs(PM_strain)), log10A_fit_pc)#, log10A_fit_strain, log10A_fit_strain_15db)
print('SNRs are (PM, SQZ, FIT, CROSS, FRAC): ', snr, snr_sqz, snr_fit_pc, snr_cross, SNR_frac)#, snr_fit_strain, snr_fit_strain_15db)
print('neg_logls are: ', neg_logl_pc)#, neg_logl_strain * np.sum(detector_sqz.total_psd/detector_nosqz.total_psd), neg_logl_strain_15db * np.sum(detector_sqz.total_psd*10**(-0.5)/detector_nosqz.total_psd))
print('Photon counts are: ', np.sum(signal_photons), np.sum(noise_photons), np.sum(noise_photons_0d1))

#A_amps = 10**log10A_fit_pc * np.max(np.abs(PM_strain)) * 10**np.random.uniform(-2, 1, size=N_samples)
A_amps = 10**log10A_fit_pc * np.max(np.abs(PM_strain)) * np.random.uniform(0, 2, size=N_samples)

weights = A_amps / 10**log10A_fit_pc / np.max(np.abs(PM_strain))

sample_indexes = np.random.choice(10000, size=N_samples, replace=False)
gammas = gamma_samples[sample_indexes]

phi0s = np.random.uniform(0, 2*np.pi, N_samples)
epsilons = np.random.normal(loc=0, scale=61, size=N_samples)

#PM_strain
observed_strain = PM_strain + gaussian_likelihood.generate_realization(detector_sqz.total_psd, frequencies)
observed_strain_15db = PM_strain + gaussian_likelihood.generate_realization(10**(-0.5) * detector_sqz.total_psd, frequencies)
observed_strain_20db = PM_strain + gaussian_likelihood.generate_realization(10**(-1) * detector_sqz.total_psd, frequencies)

for l, f0 in enumerate(f0_R1d6s):

    likelihood_array_i = np.zeros(N_samples)
    likelihood_array_i_0d01 = np.zeros(N_samples)
    likelihood_array_i_0d1 = np.zeros(N_samples)
    likelihood_array_i_0d5 = np.zeros(N_samples)
    likelihood_array_i_no_background = np.zeros(N_samples)
    likelihood_array_i_strain = np.zeros(N_samples)
    likelihood_array_i_strain_15db = np.zeros(N_samples)
    likelihood_array_i_strain_20db = np.zeros(N_samples)

    print(l, R1d6s[l], f0)

    for j in range(N_samples):

        A = A_amps[j]

        expected_photon_count_signal = LorentzianModel.generate_photon_count(
            detector_nosqz, frequencies, f0=f0 + epsilons[j], gamma=gammas[j], A=amplitude_samples[sample_indexes[j]],
            phase=phi0s[j], t0=t0s)
        expected_strain = LorentzianModel.generate_strain(
            detector_nosqz, frequencies, f0=f0 + epsilons[j], gamma=gammas[j], A=amplitude_samples[sample_indexes[j]],
            phase=phi0s[j], t0=t0s)
        expected_strain_15db = LorentzianModel.generate_strain(
            detector_nosqz, frequencies, f0=f0 + epsilons[j], gamma=gammas[j], A=amplitude_samples[sample_indexes[j]],
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
        likelihood_array_i_strain_15db[j] = gaussian_likelihood(observed_strain_15db, expected_strain_15db, 10**(-0.5) * detector_sqz.total_psd, frequencies)
        likelihood_array_i_strain_20db[j] = gaussian_likelihood(observed_strain_20db, expected_strain, 10**(-1) * detector_sqz.total_psd, frequencies)

    #, b=weights
    likelihood_event_i[l] = jax.scipy.special.logsumexp(likelihood_array_i) - jnp.log(len(likelihood_array_i))
    likelihood_event_i_0d01[l] = jax.scipy.special.logsumexp(likelihood_array_i_0d01) - jnp.log(len(likelihood_array_i_0d01))
    likelihood_event_i_0d1[l] = jax.scipy.special.logsumexp(likelihood_array_i_0d1) - jnp.log(len(likelihood_array_i_0d1))
    likelihood_event_i_0d5[l] = jax.scipy.special.logsumexp(likelihood_array_i_0d5) - jnp.log(len(likelihood_array_i_0d5))
    likelihood_event_i_no_background[l] = jax.scipy.special.logsumexp(likelihood_array_i_no_background) - jnp.log(len(likelihood_array_i_no_background))
    likelihood_event_i_strain[l] = jax.scipy.special.logsumexp(likelihood_array_i_strain) - jnp.log(len(likelihood_array_i_strain))
    likelihood_event_i_strain_15db[l] = jax.scipy.special.logsumexp(likelihood_array_i_strain_15db) - jnp.log(len(likelihood_array_i_strain_15db))
    likelihood_event_i_strain_20db[l] = jax.scipy.special.logsumexp(likelihood_array_i_strain_20db) - jnp.log(len(likelihood_array_i_strain_20db))
    # these are for computing the effective N. The loglen is included to compensate for it being in the other array
    likelihood_event_i_strainMx[l] = jnp.max(likelihood_array_i_strain) - jnp.log(len(likelihood_array_i_strain))
    likelihood_event_i_strain_15dbMx[l] = jnp.max(likelihood_array_i_strain_15db) - jnp.log(len(likelihood_array_i_strain))
    likelihood_event_i_strain_20dbMx[l] = jnp.max(likelihood_array_i_strain_20db) - jnp.log(len(likelihood_array_i_strain))

    print('PC Likelihoods are: ', likelihood_event_i[l], likelihood_event_i_0d1[l])
    print('ST Likelihoods are: ', likelihood_event_i_strain[l], likelihood_event_i_strain_15db[l])
    print(
        'ST Effective N are: ',
        np.exp(likelihood_event_i_strain[l] - likelihood_event_i_strainMx[l]),
        np.exp(likelihood_event_i_strain_15db[l] - likelihood_event_i_strain_15dbMx[l]),
    )

data = {'logls':list(likelihood_event_i),
        'logls_0d01':list(likelihood_event_i_0d01),
        'logls_0d1':list(likelihood_event_i_0d1),
        'logls_0d5':list(likelihood_event_i_0d5),
        'logls_no_background':list(likelihood_event_i_no_background),
        'logls_strain':list(likelihood_event_i_strain),
        'logls_strain_15db':list(likelihood_event_i_strain_15db),
        'logls_strain_20db':list(likelihood_event_i_strain_20db),
        'logls_strainM':list(likelihood_event_i_strainMx),
        'logls_strain_15dbM':list(likelihood_event_i_strain_15dbMx),
        'logls_strain_20dbM':list(likelihood_event_i_strain_20dbMx),
        'n_signal_photons':float(jnp.sum(signal_photons)),
        'n_noise_photons':float(jnp.sum(noise_photons)),
        'n_noise_photons_0d01':float(jnp.sum(noise_photons_0d01)),
        'n_noise_photons_0d1':float(jnp.sum(noise_photons_0d1)),
        'n_noise_photons_0d5':float(jnp.sum(noise_photons_0d5)),
        'snr':float(snr), 'snr_sqz':float(snr_sqz), 'n_exp':float(n_exp),
        'mtot':mtots, 'z':z, 'phi':phi, 'psi':psi, 'ra':ra, 'dec':dec, 'iota':iota, 'f0_fit':f0_fit,
        'fpeak_fit':fpeak_fit, 'gamma_fit':gamma_fit, 'log10A_fit_pc':float(np.log10(10**log10A_fit_pc * np.max(np.abs(PM_strain)))),
        'snr_fit_pc':float(snr_fit_pc), 'phase_fit':phase_fit}

with open(fname, 'w') as f:
    json.dump(data, f)

import GWPhotonCounting
import jax.numpy as jnp
import numpy as np
import bilby


from corner import corner
import matplotlib.pyplot as plt
from tqdm import tqdm

from astropy.cosmology import Planck18
import astropy.units as u

from scipy.optimize import minimize

import jax
jax.config.update("jax_enable_x64", True)

frequencies = jnp.sort(jnp.fft.fftfreq(2**13, d=1/1e4))

# Setting up the two detectors to compare the
detector_nosqz = GWPhotonCounting.detector.Detector(
    frequencies, '/home/mcculler/projects/GWPhotonCounting/examples/data/CE_shot_psd_nosqz.csv',
    '/home/mcculler/projects/GWPhotonCounting/examples/data/CE_classical_psd.csv',
    gamma=80, random_seed=1632, N_frequency_spaces=17*2, N_time_spaces=4)

detector_sqz = GWPhotonCounting.detector.Detector(
    frequencies, '/home/mcculler/projects/GWPhotonCounting/examples/data/CE_total_psd_sqz.csv', None,
    gamma=80, random_seed=1632, N_frequency_spaces=17*2, N_time_spaces=4)

zmax = 10
zinterp = np.expm1(np.linspace(np.log1p(0), np.log1p(zmax), 2000))
dVdzdt_interp = 4*np.pi*Planck18.differential_comoving_volume(zinterp).to(u.Gpc**3/u.sr).value/(1+zinterp)

KNNModel = GWPhotonCounting.signal.PostMergerKNN(knn_file_path='/home/mcculler/photon_counting/apr4_knn_gw_model_2024/KNN_Models/APR4-knn_model-N100')
LorentzianModel = GWPhotonCounting.signal.PostMergerLorentzian()

gaussian_likelihood = GWPhotonCounting.distributions.GaussianStrainLikelihood()

def sample_redshift(n):
    pdf_red = ((1+zinterp)**2.7)/(1+((1+zinterp)/2.9)**5.6) * dVdzdt_interp
    cum_sum_red = np.cumsum(pdf_red)/np.sum(pdf_red)

    return np.interp(np.random.uniform(size=n), cum_sum_red, zinterp)

def neg_logl_cost_function(x0, strain, frequencies):

    f0, gamma, log10A, phase = x0

    t0s = jnp.linspace(-0.02, 0.02, 10)

    # Calculate the expected strain
    lorentzian_strain = LorentzianModel.generate_strain(
        detector_nosqz, frequencies, f0=f0, gamma=gamma, A=10**log10A, phase=phase, t0=t0s)

    # Calculate the log-likelihood
    logl = gaussian_likelihood(strain, lorentzian_strain, jnp.ones(len(frequencies))*1e-55,frequencies)

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


# Adding samples to the dataset
bns_pm_dataset = []

for N in tqdm(range(int(1.7e4))):
    m1 = bilby.gw.prior.Uniform(1.2,1.4).sample(1)[0]
    m2 = bilby.gw.prior.Uniform(1.2,1.4).sample(1)[0]
    mtots = m1+m2

    z = sample_redshift(1)[0]

    phi = bilby.gw.prior.Uniform(0,2*np.pi).sample(1)[0]
    psi = bilby.gw.prior.Uniform(0,np.pi).sample(1)[0]

    ra = bilby.core.prior.Uniform(0,2*np.pi).sample(1)[0]
    dec = bilby.core.prior.Cosine().sample(1)[0]

    iota = bilby.core.prior.Sine().sample(1)[0]

    PM_strain = KNNModel.generate_strain(detector_nosqz, frequencies, mtots, phi, z, ra, dec, iota, psi)

    prior_fit = None
    for i in range(5):
        params_0 = [np.abs(frequencies[np.argmax(np.abs(PM_strain))]), np.random.uniform(1,70), jnp.log10(np.max(np.abs(PM_strain))), np.random.uniform(0,2*np.pi)]

        minimize_result = minimize(neg_logl_cost_function,
                                    x0=params_0,
                                    args=(PM_strain, frequencies))

        f0_fit, gamma_fit, log10A_fit, phase_fit = minimize_result.x

        which = neg_logl_cost_function_split(minimize_result.x, PM_strain, frequencies)
        t0_fit, logl = which

        print("FIT: ", log10A_fit)
        if gamma_fit < 0:
            gamma_fit = np.abs(gamma_fit)
            phase_fit += np.pi

        phase_fit = phase_fit % (2*np.pi)

        snr_CE = detector_nosqz.calculate_optimal_snr(PM_strain, frequencies)
        SNR_CEsilica = detector_sqz.calculate_optimal_snr(PM_strain, frequencies)

        lorentzian_strain = LorentzianModel.generate_strain(
            detector_nosqz, frequencies, f0=f0_fit, gamma=gamma_fit, A=10**log10A_fit, phase=phase_fit, t0=t0_fit)[0, :]

        snr_cross = detector_nosqz.calculate_inner_product(PM_strain, lorentzian_strain, frequencies)
        snr_lorentzian = detector_nosqz.calculate_optimal_snr(lorentzian_strain, frequencies)
        SNR_frac = snr_cross**2 / snr_lorentzian / snr_CE
        print("SNR, SNR_L, SNR_X, fraction_overlap: ", snr_CE, snr_lorentzian, snr_cross, SNR_frac)

        if prior_fit is not None and prior_fit[0] > SNR_frac:
            SNR_frac, f0_fit, gamma_fit, log10A_fit, phase_fit, t0_fit = prior_fit
            print('SNR fraction not improved, try another fit')
        elif SNR_frac > .6:
             # get out of the fit if it was a decent one
             break
        else:
            prior_fit = (SNR_frac, f0_fit, gamma_fit, log10A_fit, phase_fit, t0_fit)
            print('try another fit')

    if log10A_fit < -20:
        if gamma_fit > 1e-5:
            if log10A_fit > -29:
                bns_pm_dataset.append([mtots, z, phi, psi, ra, dec, iota, f0_fit, gamma_fit, 10**log10A_fit, phase_fit, t0_fit, float(snr_CE), float(SNR_CEsilica), float(snr_lorentzian), float(snr_cross), float(SNR_frac)])

bns_pm_dataset = np.array(bns_pm_dataset)
np.savetxt('bns_pm_dataset_MLE_260214.dat', bns_pm_dataset,
           header='mtots z phi psi ra dec iota f0_fit gamma_fit A_fit phase_fit t0 snr snr_sqz snr_lorentzian snr_cross snr_frac')

# Plotting the distribution of signals
# corner(np.array([bns_pm_dataset[:,-3], bns_pm_dataset[:,-2], bns_pm_dataset[:,-4], bns_pm_dataset[:,1], bns_pm_dataset[:,0]]).T, plot_contours=True, plot_density=False, levels=[0.5,0.9], plot_datapoints=False,
#        labels=[r'SNR', r'SNR sqz', r'fpeak [Hz]', r'redshift',  r'Mtot'])
# plt.savefig('corner_plot.pdf', bbox_inches='tight')

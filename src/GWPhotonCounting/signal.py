import jax.numpy as jnp
import numpy as np
import jax
import pickle as pkl
import bilby
from scipy.interpolate import interp1d
from astropy.cosmology import Planck18

from scipy.signal import savgol_filter
from scipy.signal import find_peaks

from .utils import phase_shift, lorentzian_complex

# Constants
CU_to_ms = 0.004925 # Geom to ms

class BasePhotonCountingSignal:

    def __init__(self):
        pass

    def generate_strain(self, detector, frequencies, **kwargs):
        """This needs to be overwritten by the subclass."""
        pass

    def generate_photon_count(self, detector, frequencies, **kwargs):

        strain = self.generate_strain(detector, frequencies, **kwargs)

        photon_count_expectation = detector.calculate_signal_photon_expectation(strain, frequencies)

        return photon_count_expectation

class PostMergerKNN(BasePhotonCountingSignal):
    '''
    Class using the KNN post-merger model from XXX to generate the strain signal. We'll convert into photons
    using the detector class later.
    '''

    def __init__(self, ti=0, tf=20, dt=5, knn_file_path='/home/ethan.payne/code_libraries/apr4_knn_gw_model_2024/KNN_Models/APR4-knn_model-N100'):
        with open(knn_file_path, 'rb') as file:
            self.KNN_model = pkl.load(file)

        self.dt_ms = dt * CU_to_ms
        self.time_array = jnp.arange(ti, tf, self.dt_ms)

    def generate_plus_cross_strain(self, frequencies, mtot, phi0, z):

        mtot_arr = jnp.zeros(self.time_array.size) + mtot

        # Generatinging the hplus signal from the KKN model - note the complicated factor for scaling to the correct distance
        hplus = self.KNN_model.predict(jnp.transpose(jnp.vstack((mtot_arr, self.time_array)))) * (1+z)/Planck18.luminosity_distance(z).value / 3.085677581491367e+22 * mtot * 2e30 * 6.67e-11/(3e8)**2


        hplus =  phase_shift(hplus, angle = 2*phi0)
        hcross = phase_shift(hplus, angle = 2*phi0+jnp.pi/2)

        i_array = np.array(range(len(self.time_array)))
        n_taper_start = (find_peaks(savgol_filter(np.abs(hplus)[:1000], 25,3))[0][4] + find_peaks(savgol_filter(np.abs(hcross)[:1000], 25,3))[0][4])/2

        with np.errstate(divide='ignore'):
            z_start = (n_taper_start/i_array) + n_taper_start/(i_array - n_taper_start)
        sigma = 1/(np.exp(z_start)+1) *(i_array < n_taper_start) + 1*(i_array >= n_taper_start)

        hp_fd, freq_array = bilby.core.utils.nfft(hplus * sigma, sampling_frequency=1e3/(self.dt_ms))
        hc_fd, freq_array = bilby.core.utils.nfft(hcross * sigma, sampling_frequency=1e3/(self.dt_ms))

        return interp1d(jnp.concatenate((freq_array/(1+z), -freq_array[1:]/(1+z))), jnp.concatenate((hp_fd, jnp.conj(hp_fd[1:]))), kind='cubic', bounds_error=False, fill_value=0)(frequencies), \
            interp1d(jnp.concatenate((freq_array/(1+z), -freq_array[1:]/(1+z))), jnp.concatenate((hc_fd, jnp.conj(hc_fd[1:]))), kind='cubic', bounds_error=False, fill_value=0)(frequencies)

    def generate_strain(self, detector, frequencies, mtot, phi0, z, ra, dec, iota, psi):

        hp, hc = self.generate_plus_cross_strain(frequencies, mtot, phi0, z)

        # Using Bilby to get the antenna response function for the frequency dependent detector behavior
        h = detector.ifo.get_detector_response({'plus':np.array(hp * (1+jnp.cos(iota)**2)/2), 'cross':np.array(hc * jnp.cos(iota))},
                                                {'ra':ra, 'dec':dec, 'psi':psi, 'iota':iota, 'geocent_time':0},
                                                frequencies=np.array(frequencies, dtype=np.float64))

        return jnp.nan_to_num(h)

class PostMergerLorentzian(BasePhotonCountingSignal):

    def __init__(self):
        pass

    def generate_strain(self, detector, frequencies, f0, gamma, A, phase, t0):
        t0 = jnp.asarray(t0)
        return ( lorentzian_complex(frequencies.reshape(1, -1), f0, gamma, A, phase - 2 * jnp.pi * t0.reshape(-1, 1) * f0)
                * jnp.exp(-2j * jnp.pi * t0.reshape(-1, 1) * frequencies.reshape(1, -1))
                )





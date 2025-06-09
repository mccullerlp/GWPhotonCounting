from .utils import lorentzian_complex, load_and_interpolate_data, gram_schmidt
import jax.numpy as jnp
import jax
import numpy as np

import bilby


class Detector:
    def __init__(self, frequencies, shot_noise_psd, classical_noise_psd, detector_asd=False, ifo_name='CE',
                 base_filter_function=lorentzian_complex, N_frequency_spaces=15, N_time_spaces=10,
                 minimum_frequency=1.5e3, maximum_frequency=4e3, maximum_duration=4e-2,
                 random_seed=1234, gaussian_noise=False, **kwargs):
        # Load and interpolate shot noise PSD data
        if detector_asd is False:
            self.shot_noise_psd = load_and_interpolate_data(shot_noise_psd, frequencies)
        else:
            self.shot_noise_psd = load_and_interpolate_data(shot_noise_psd, frequencies)**2

        # Load and interpolate classical noise quanta data, then transform it back into a PSD
        if classical_noise_psd is not None:
            self.classical_noise_psd = load_and_interpolate_data(classical_noise_psd, frequencies)
        else:
            self.classical_noise_psd = np.zeros_like(self.shot_noise_psd)

        # Generate the total PSD
        self.total_psd = self.shot_noise_psd + self.classical_noise_psd
        
        # Setting up the spacing of the temporal mode filters
        self.N_total_filters = int(2 * maximum_duration * (maximum_frequency - minimum_frequency))
        self.N_time_spaces = N_time_spaces
        self.N_frequency_spaces = N_frequency_spaces

        print('N_total_filters for Nyquist:', self.N_total_filters)
        print('N_total_filters from user:', self.N_time_spaces*self.N_frequency_spaces * 2)
        
        self.f0_values = jnp.linspace(minimum_frequency, maximum_frequency, N_frequency_spaces)
        self.t0_values = jnp.linspace(-maximum_duration/2, maximum_duration/2, self.N_time_spaces)

        # Construct the arrays of filter functions
        self.filter_functions, self.filter_labels = self._construct_filter_functions(
            frequencies, base_filter_function, random_seed, **kwargs
        )
        
        # Setting up the interferometer
        self.ifo = bilby.gw.detector.FrequencyDependentInterferometer.from_name(ifo_name, time_dependent=False)
        self.ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity.from_power_spectral_density_array(
            frequencies, np.array(self.total_psd))
        
        if gaussian_noise is False:
            self.ifo.set_strain_data_from_zero_noise(sampling_frequency=2*frequencies[-1], duration=1/(frequencies[1]-frequencies[0]), start_time=0)
        elif gaussian_noise is True:
            self.ifo.set_strain_data_from_power_spectral_density(sampling_frequency=2*frequencies[-1], duration=1/(frequencies[1]-frequencies[0]), start_time=0)

        # Compute the number of expected photons from the noise background here
        self.noise_photon_expectation = self.calculate_noise_photon_expectation(frequencies)

    def _construct_filter_functions(self, frequencies, filter_function,
                                    random_seed, **kwargs):
        """Helper method to construct filter functions."""
        
        
        filter_functions_at_t0 = jnp.concatenate([
            jnp.array([filter_function(frequencies, f0=f0, A=1, phase=0, **kwargs) for f0 in self.f0_values]),
            jnp.array([filter_function(frequencies, f0=f0, A=1, phase=jnp.pi/2, **kwargs) for f0 in self.f0_values])])
        
        filter_labels_f0_t0 = [(f0, t0) for f0 in self.f0_values for t0 in self.t0_values]

        filter_labels = []

        for phase in (0, jnp.pi/2):
            filter_labels += [(float(f0), float(t0), phase) for f0, t0 in filter_labels_f0_t0]

        time_delays = jnp.exp(-2j * jnp.pi * jnp.outer(self.t0_values, frequencies))
        
        filter_functions = jnp.einsum('ij, kj -> ikj', filter_functions_at_t0, time_delays).reshape(-1, len(frequencies))
        
        key = jax.random.PRNGKey(random_seed)
        shuffled_indices = jax.random.permutation(key, self.N_total_filters)
        orthonormalized_filters = gram_schmidt(filter_functions[shuffled_indices]/jnp.sqrt(self.shot_noise_psd))/jnp.sqrt(frequencies[1] - frequencies[0])

        shuffled_filter_labels = []
        for i in shuffled_indices:
            shuffled_filter_labels.append(filter_labels[i])

        return orthonormalized_filters, shuffled_filter_labels
    
    def calculate_signal_photon_expectation(self, strain, frequencies):
        
        quanta_amplitude = strain/jnp.sqrt(2 * self.shot_noise_psd)

        integral = jnp.einsum('...k, jk -> ...j', jnp.conj(quanta_amplitude), self.filter_functions * jnp.diff(frequencies, append=frequencies[-1]))

        signal_photon_expectation = jnp.abs(integral)**2
        
        return signal_photon_expectation

    def calculate_noise_photon_expectation(self, frequencies):
        noise_photon_expectation = jnp.zeros(self.N_total_filters, dtype=float)

        noise_quanta_psd = self.classical_noise_psd/(2 * self.shot_noise_psd)

        for i in range(self.N_total_filters):
            noise_photon_expectation = noise_photon_expectation.at[i].set(
                jnp.sum(noise_quanta_psd * jnp.abs(self.filter_functions[i])**2 * jnp.diff(frequencies, append=frequencies[-1]))
            )

        return noise_photon_expectation
    
    def calculate_optimal_snr(self, strain, frequencies, fmin = 100, fmax = 4e3):
        mask1 = (frequencies >= fmin) & (frequencies <= fmax)
        mask2 = (frequencies <= -fmin) & (frequencies >= -fmax)
        mask = mask1 | mask2
        optimal_snr = (2*np.real(np.sum(strain[mask] * np.conj(strain[mask])/self.total_psd[mask]))*(frequencies[1]-frequencies[0]))**0.5
        return optimal_snr

from .utils import lorentzian_complex, load_and_interpolate_data, gram_schmidt
import jax.numpy as jnp
import jax
import numpy as np

import bilby


class Detector:
    def __init__(self, frequencies, shot_noise_psd, classical_noise_psd, detector_asd=False, ifo_name='CE',
                 base_filter_function=lorentzian_complex, N_frequency_spaces=15, N_time_spaces=10,
                 minimum_frequency=.6e3, maximum_frequency=4e3, maximum_duration=4e-2,
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
        
        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency
        # Setting up the spacing of the temporal mode filters
        self.N_nyquist = int(2 * maximum_duration * (maximum_frequency - minimum_frequency))
        self.N_time_spaces = N_time_spaces
        self.N_frequency_spaces = N_frequency_spaces
        self.N_total_filters = self.N_time_spaces*self.N_frequency_spaces * 2

        print('N_total_filters for Nyquist:', self.N_nyquist)
        print('N_total_filters from user:', self.N_time_spaces*self.N_frequency_spaces * 2)
        
        self.f0_values = jnp.geomspace(minimum_frequency, maximum_frequency, N_frequency_spaces)
        self.t0_values = jnp.linspace(-maximum_duration * 2/5, maximum_duration * 3/5, self.N_time_spaces)

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
        # for testing inefficiency
        # filter_functions_at_t0 = filter_functions_at_t0.conjugate()
        
        filter_labels_f0_t0 = [(f0, t0) for f0 in self.f0_values for t0 in self.t0_values]

        filter_labels = []

        for phase in (0, jnp.pi/2):
            filter_labels += [(float(f0), float(t0), phase) for f0, t0 in filter_labels_f0_t0]

        time_delays = jnp.exp(-2j * jnp.pi * jnp.outer(self.t0_values, frequencies))
        
        filter_functions = jnp.einsum('ij, kj -> ikj', filter_functions_at_t0, time_delays).reshape(-1, len(frequencies))
        self.filter_function_orig = filter_functions
        
        key = jax.random.PRNGKey(random_seed)
        shuffled_indices = jax.random.permutation(key, self.N_total_filters)
        orthonormalized_filters = gram_schmidt(filter_functions[shuffled_indices]/jnp.sqrt(self.shot_noise_psd))/jnp.sqrt(frequencies[1] - frequencies[0])
        dF = frequencies[1] - frequencies[0]

        print("FSHAPE", orthonormalized_filters.shape)
        efficiencies = jnp.einsum('ij, ji -> i',orthonormalized_filters, (orthonormalized_filters.T.conjugate())) * dF
        print('eff', np.average(efficiencies.real))
        select = (frequencies > 500)
        efficiencies15 = (jnp.einsum('ij, ji -> i',orthonormalized_filters[:, select], (orthonormalized_filters[:, select].T.conjugate())) * dF).real * 2
        print('eff15', np.average(efficiencies15))
        shuffled_filter_labels = []
        for i in shuffled_indices:
            shuffled_filter_labels.append(filter_labels[i])

        return orthonormalized_filters, shuffled_filter_labels
    
    def calculate_signal_photon_expectation(self, strain, frequencies):
        
        quanta_amplitude = strain/jnp.sqrt(2 * self.shot_noise_psd)

        integral = jnp.einsum('...k, jk -> ...j', jnp.conj(quanta_amplitude), self.filter_functions * jnp.diff(frequencies, append=frequencies[-1]))

        signal_photon_expectation = jnp.abs(integral)**2

        return signal_photon_expectation

    def calculate_signal_photon_expectation_self(self, strain, frequencies):
        quanta_amplitude = strain/jnp.sqrt(2 * self.shot_noise_psd)

        dF = frequencies[1] - frequencies[0]
        norm = np.sqrt(jnp.einsum('...j, j -> ...',quanta_amplitude, (quanta_amplitude.T.conjugate())) * dF)
        integral_self = jnp.einsum('...k, k -> ...', jnp.conj(quanta_amplitude), quanta_amplitude * jnp.diff(frequencies, append=frequencies[-1])) / norm
        select = (frequencies < self.maximum_frequency + 100) & (frequencies > self.minimum_frequency - 100)

        # now get the value restricted to positive frequency band
        quanta_amplitude = quanta_amplitude[select]
        frequencies = frequencies[select]

        integral_self2 = (jnp.einsum('...k, k -> ...', jnp.conj(quanta_amplitude), quanta_amplitude * jnp.diff(frequencies, append=frequencies[-1])) / norm).real * 2
        return np.abs(integral_self)**2, np.abs(integral_self2)**2

    def calculate_noise_photon_expectation(self, frequencies):
        noise_photon_expectation = jnp.zeros(self.N_total_filters, dtype=float)

        noise_quanta_psd = self.classical_noise_psd/(4 * self.shot_noise_psd)

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

    def calculate_inner_product(self, strain1, strain2, frequencies, fmin = 100, fmax = 4e3):
        mask1 = (frequencies >= fmin) & (frequencies <= fmax)
        mask2 = (frequencies <= -fmin) & (frequencies >= -fmax)
        mask = mask1 | mask2
        optimal_snr = (2*np.abs(np.sum(strain1[mask] * np.conj(strain2[mask])/self.total_psd[mask]))*(frequencies[1]-frequencies[0]))**0.5
        return optimal_snr

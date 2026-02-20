import jax.numpy as jnp
import jax
import numpy as np
from numpyro.distributions import Poisson, Geometric
from jax import random
from jax.scipy.special import factorial
from scipy.interpolate import interp1d

class BaseLikelihood():

    def __init__(self):
        pass

    def generate_realization(self, data):
        pass

    def log_likelihood_individual(self, observed_data, model_data):
        """
        Individual log likelihood for each temporal mode filter
        """
        pass

    def log_likelihood(self, observed_data, model_data):
        return jnp.sum(
            self.log_likelihood_individual(observed_data, model_data),
            axis=1)

    def __call__(self, observed_data, model_data):
        """
        Note that this includes the time marginalization by summing
        over the axis
        """
        return jax.scipy.special.logsumexp(self.log_likelihood(observed_data, model_data)) - jnp.log(model_data.shape[0])

class PoissonPhotonLikelihood(BaseLikelihood):

    def generate_realization(self, data):
        return Poisson(data).sample(random.PRNGKey(np.random.randint(0,100000)))

    def log_likelihood_individual(self, observed_data, model_data):
        return jnp.atleast_2d(Poisson(model_data).log_prob(observed_data))


class GeometricPhotonLikelihood(BaseLikelihood):

    def generate_realization(self, data):
        '''
        Note that the data array is expected photon count not a probability
        '''
        return Geometric(1/(data+1)).sample(random.PRNGKey(np.random.randint(0,100000)))

    def log_likelihood_individual(self, observed_data, model_data):
        return jnp.atleast_2d(Geometric(1/(model_data+1)).log_prob(observed_data))

class PhaseQuadraturePhotonLikelihood(BaseLikelihood):

    def __init__(self, n_max=3):
        self.ns = jnp.linspace(0,n_max, n_max+1)

    def calculate_probabilities(self, data):

        prefactor = factorial(2 * self.ns)/2**self.ns/factorial(self.ns)**2

        return prefactor * jnp.power(jnp.outer(data, jnp.ones(len(self.ns))),self.ns)/\
            jnp.power(2*jnp.outer(data, jnp.ones(len(self.ns)))+1,self.ns + 0.5)

    def generate_realization(self, data):

        # Has dimensions of N_filters x photon counts + 1
        probabilities = self.calculate_probabilities(data)

        val = jnp.outer(
            random.uniform(random.PRNGKey(np.random.randint(0,100000)), shape=(len(data),)),
            jnp.ones(self.ns.shape[0]))

        count_idx = jnp.argmax(val < jnp.cumsum(probabilities, axis=1), axis=1)

        return self.ns[count_idx]

    def log_likelihood_individual(self, observed_data, model_data):

        prefactor = jnp.log(factorial(2 * observed_data)/2**observed_data/factorial(observed_data)**2)

        return jnp.atleast_2d(prefactor + jnp.log( jnp.power(model_data,observed_data)/
            jnp.power(2*model_data+1, observed_data + 0.5)))


class MixturePhotonLikelihood(object):

    def __init__(self, signal_model, noise_model):
        self.signal_model = signal_model
        self.noise_model = noise_model

    def generate_realization(self, signal_data, noise_data):

        signal_photons = self.signal_model.generate_realization(signal_data)
        noise_photons = self.noise_model.generate_realization(noise_data)

        return signal_photons + noise_photons, signal_photons, noise_photons

    def log_likelihood_individual_inner_term(
            self, m, observed_data, model_data_signal, model_data_noise):

        signal_log_likelihood = \
            self.signal_model.log_likelihood_individual(observed_data - m, model_data_signal.T)

        noise_log_likelihood = \
            self.noise_model.log_likelihood_individual(m, model_data_noise)

        return signal_log_likelihood + noise_log_likelihood

    def log_likelihood_individual(self, observed_data, model_data_signal, model_data_noise):

        m_max_counter = 2
        m_array = jnp.linspace(0,m_max_counter, m_max_counter+1, dtype=int)

        log_likelihood_vmap_indiv_inner = jax.vmap(self.log_likelihood_individual_inner_term, in_axes=(0,None, None, None))
        log_likelihood_vmap_indiv = jax.vmap(log_likelihood_vmap_indiv_inner, in_axes=(None,0,0, 0))

        log_likelihood_m_individual = log_likelihood_vmap_indiv(m_array, observed_data, model_data_signal.T, model_data_noise)

        log_likelihood_indiv = jnp.atleast_2d(jax.scipy.special.logsumexp(log_likelihood_m_individual, axis=1))[:,0,:].T

        return log_likelihood_indiv

    def log_likelihood(self, observed_data, model_data_signal, model_data_noise):
        return jnp.sum(
            self.log_likelihood_individual(observed_data, model_data_signal, model_data_noise),
            axis=1)

    def __call__(self, observed_data, model_data_signal, model_data_noise):
        """
        Note that this includes the time marginalization by summing
        over the axis
        """
        return jax.scipy.special.logsumexp(self.log_likelihood(observed_data, model_data_signal, model_data_noise)) - jnp.log(model_data_signal.shape[0])


class GaussianStrainLikelihood():
    # Standard Gaussian homodyne readout strain likelihood

    def generate_realization(self, psd_data, frequencies):
        '''
        Note that the data here should be the PSD
        '''

        # duration = 2**13/10**4

        # TODO make this work on non-linear frequencies array
        dF = frequencies[1] - frequencies[0]
        duration = 1/dF
        norm1 = 0.5 *duration**0.5 #(duration/2)**0.5

        np.random.seed()
        re1, im1 = norm1*random.normal(random.PRNGKey(np.random.randint(0,100000)), shape=(2, len(frequencies[frequencies>=0])))
        white_noise_pos = (re1 + 1j * im1)


        white_noise_freq = np.concatenate([[0], np.conj(white_noise_pos[1:])[::-1], [re1[0]], white_noise_pos[1:]])


        # white_noise_freq = np.zeros_like(frequencies, dtype=np.complex128)
        # white_noise_freq[frequencies >= 0] = white_noise_pos
        # white_noise_freq[frequencies < 0][1:] = np.conjugate(white_noise_pos)[::-1][:-1]

        # white_noise_freq[len(frequencies)//2] = white_noise_freq[len(frequencies)//2].real  # ensure real at DC frequency
        # white_noise_freq[0] = 0  # ensure zero at Nyquist

        return white_noise_freq * psd_data ** 0.5

    def log_likelihood(self, observed_data, model_data, psd_data, frequencies):

        residual = observed_data - model_data

        # note no factor of two since the psd is defined in negative and positive frequencies
        # the len(frequencies) is to remove the "average" likelihood
        return (
            # len(frequencies)/2
            - np.sum(np.abs(residual)**2 / psd_data, axis=1) * (frequencies[1]-frequencies[0])
            )
        #return - jnp.real(jnp.einsum('ij, ij -> i', residual,jnp.conj(residual)/psd_data)) * (frequencies[1]-frequencies[0]) # note no factor of two since the psd is defined in negative and positive frequencies

    def __call__(self, observed_data, model_data, psd_data, frequencies):
        """
        Note that this includes the time marginalization by summing
        over the axis
        """
        return jax.scipy.special.logsumexp(self.log_likelihood(observed_data, model_data, psd_data, frequencies)) - jnp.log(model_data.shape[0])

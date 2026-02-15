import numpy as np
import jax.numpy as jnp
from scipy.interpolate import interp1d

def load_data(file_path):
        try:
                data = np.loadtxt(file_path, delimiter=',')
        except:
                data = np.loadtxt(file_path)
        return data[:, 0], data[:, 1]

def load_and_interpolate_data(data_path, frequencies):
        """Helper method to load and interpolate data."""
        data = load_data(data_path)
        return interp1d(data[0], data[1], bounds_error=False, fill_value=1)(frequencies)*interp1d(data[0], data[1], bounds_error=False, fill_value=1)(-frequencies)

def lorentzian_complex(f, f0, gamma, A, phase):
    return A * gamma / (2 * jnp.sqrt(2) * jnp.pi**(3/2)) * \
        (jnp.exp(-1j * phase)/(gamma**2 + (f - f0)**2) +
         jnp.exp(1j * phase)/(gamma**2 + (f + f0)**2))

def lorentzian_complex(f, f0, gamma, A, phase):
    """
    gamma is the FWHM

    TODO fix normalization
    """
    return ( A * gamma *
         ( jnp.exp(-1j * phase)/(gamma - 1j*(f0 - f))
           + jnp.exp(1j * phase)/(gamma + 1j*(f0 + f))
         ))

def gram_schmidt(X):
    Q, R = np.linalg.qr(X.T)
    return Q.T


def phase_shift(signal, angle = jnp.pi):
        N = signal.shape[0]
        y = jnp.zeros(3*N)

        y = y.at[N:2*N].set(signal)

        rotation       = jnp.exp(1j * angle)
        signal_shifted = jnp.fft.irfft(jnp.fft.rfft(y, n=y.shape[0])*rotation, n=y.shape[0])

        return signal_shifted[N:2*N]

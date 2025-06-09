import numpyro
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az
import os
import json
from tqdm import tqdm

from .signal import PostMergerLorentzian

def save_analyses(filename, fit_pc=None, fit_pc_no_background=None, fit_strain=None, outdir='results/', **kwargs):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    json_dict = {}
    
    # Checking if a specific fit is included and if so to upload it to the file and save the .nc file
    if fit_pc is not None:
        filename_pc = outdir + '/' + filename + '_pc.nc'
        fit_pc.to_netcdf(filename_pc)

        json_dict['filename_pc'] = filename_pc

    if fit_pc_no_background is not None:
        filename_pc_no_background = outdir + '/' + filename + '_pc_no_background.nc'
        fit_pc_no_background.to_netcdf(filename_pc_no_background)

        json_dict['filename_pc_no_background'] = filename_pc_no_background

    if fit_strain is not None:
        filename_strain = outdir + '/' + filename + '_strain.nc'
        fit_strain.to_netcdf(filename_strain)

        json_dict['filename_strain'] = filename_strain

    # Adding any additional information to the json file regarding the post-merger model that was simulated
    for key in kwargs.keys():
        json_dict[key] = kwargs[key]

    with open(outdir + '/' + filename + '.json', 'w') as f:
        json.dump(json_dict, f)

class BaseInference():

    def __init__(self, detector, frequencies, likelihood, include_background=True):
        self.detector = detector
        self.frequencies = frequencies
        self.include_background = include_background
        self.likelihood = likelihood


class PhotonCountingInference(BaseInference):

    def run(self, data, num_samples=1000, num_warmup=1000, num_chains=2, f0min=1e2, f0max=4e3, time_reconstruction=True, amplitude=None, noise_scale=1, **kwargs):
        
        LorentzianModel = PostMergerLorentzian()

        # Defining the model
        def model(data_model):

            if amplitude is None:
                A = numpyro.sample('A', dist.Uniform(0,1e-19))
                f0 = numpyro.sample('f0', dist.Uniform(f0min, f0max))
                gamma = numpyro.sample('gamma', dist.Uniform(0,400))
                phase = numpyro.sample('phase', dist.Uniform(0,2*jnp.pi)) 
                t0s = jnp.linspace(-0.02, 0.02, 100)
            
                expected_photon_count_signal = LorentzianModel.generate_photon_count(self.detector, self.frequencies, f0=f0, gamma=gamma, A=A, phase=phase, t0=t0s)

            else:
                
                f0 = numpyro.sample('f0', dist.Uniform(f0min, f0max))
                gamma = numpyro.sample('gamma', dist.Uniform(0,400))
                phase = numpyro.sample('phase', dist.Uniform(0,2*jnp.pi)) 
                t0s = jnp.linspace(-0.02, 0.02, 100)

                A = jnp.interp(f0, self.frequencies, amplitude)
            
                expected_photon_count_signal = LorentzianModel.generate_photon_count(self.detector, self.frequencies, f0=f0, gamma=gamma, A=A, phase=phase, t0=t0s)

            if self.include_background:
                log_likelihood_postmerger = self.likelihood(
                    data_model, expected_photon_count_signal, noise_scale * self.detector.noise_photon_expectation)
                
            else:
                log_likelihood_postmerger = self.likelihood(
                    data_model, expected_photon_count_signal)
            
            numpyro.factor('log_likelihood', log_likelihood_postmerger)

        # Set up running the NUTS sampler now
        kernel = NUTS(model, target_accept_prob=0.8)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
        mcmc.run(jax.random.PRNGKey(np.random.randint(1<<32)), data)

        fit = az.from_numpyro(mcmc)

        if time_reconstruction:
            # Marginalize back over the time axis
            t0s = jnp.linspace(-0.02, 0.02, 1000)

            resampled_t0_vals = jnp.zeros((num_chains, num_samples))

            for n_chain in range(num_chains):
                for n_sample in tqdm(range(num_samples)):
                    expected_photon_count_signal = LorentzianModel.generate_photon_count(self.detector, self.frequencies, f0=fit.posterior['f0'].values[n_chain, n_sample], 
                                                                            gamma=fit.posterior['gamma'].values[n_chain, n_sample], 
                                                                            A=fit.posterior['A'].values[n_chain, n_sample], 
                                                                            phase=fit.posterior['phase'].values[n_chain, n_sample], t0=t0s)
                    
                    if self.include_background:
                        log_likelihood_postmerger = self.likelihood.log_likelihood(
                            data, expected_photon_count_signal, noise_scale * self.detector.noise_photon_expectation)
                        
                    else:
                        log_likelihood_postmerger = self.likelihood.log_likelihood(
                            data, expected_photon_count_signal)

                    cdf_dist = jnp.cumsum(jnp.exp(log_likelihood_postmerger - jnp.max(log_likelihood_postmerger)))/jnp.sum(jnp.exp(log_likelihood_postmerger - jnp.max(log_likelihood_postmerger)))

                    resampled_t0_vals = resampled_t0_vals.at[n_chain, n_sample].set(jnp.interp(np.random.uniform(), cdf_dist, t0s))

            fit.posterior['t0'] = (('chain','draw'),np.array(resampled_t0_vals))

        return fit


class StrainInference(BaseInference):

    def run(self, data, num_samples=1000, num_warmup=1000, num_chains=2, f0min=1e2, f0max=4e3, time_reconstruction=True, amplitude=None, noise_scale=1, **kwargs):

        LorentzianModel = PostMergerLorentzian()
        
        # Defining the model
        def model(data_model):
            if amplitude is None:
                A = numpyro.sample('A', dist.Uniform(0,1e-19))
                f0 = numpyro.sample('f0', dist.Uniform(f0min, f0max))
                gamma = numpyro.sample('gamma', dist.Uniform(0,400))
                phase = numpyro.sample('phase', dist.Uniform(0,2*jnp.pi))
                t0s = jnp.linspace(-0.02, 0.02, 100) #+ numpyro.sample('tjitter', dist.Uniform(0,0.04/100))
            
                expected_strain_signal = LorentzianModel.generate_strain(self.detector, self.frequencies, f0=f0, gamma=gamma, A=A, phase=phase, t0=t0s)

            else:
                f0 = numpyro.sample('f0', dist.Uniform(f0min, f0max))
                gamma = numpyro.sample('gamma', dist.Uniform(0,400))
                phase = numpyro.sample('phase', dist.Uniform(0,2*jnp.pi)) 
                t0s = jnp.linspace(-0.02, 0.02, 100)

                A = jnp.interp(f0, self.frequencies, amplitude)
            
                expected_strain_signal = LorentzianModel.generate_strain(self.detector, self.frequencies, f0=f0, gamma=gamma, A=A, phase=phase, t0=t0s)

            log_likelihood_postmerger = self.likelihood(data_model, expected_strain_signal, noise_scale * self.detector.shot_noise_psd + self.detector.classical_noise_psd, self.frequencies)
            
            numpyro.factor('log_likelihood', log_likelihood_postmerger)

        # Set up running the NUTS sampler now
        kernel = NUTS(model, target_accept_prob=0.8)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
        mcmc.run(jax.random.PRNGKey(np.random.randint(1<<32)), data)

        fit = az.from_numpyro(mcmc)

        if time_reconstruction:
            # Marginalize back over the time axis
            t0s = jnp.linspace(-0.02, 0.02, 1000)

            resampled_t0_vals = jnp.zeros((num_chains, num_samples))

            for n_chain in range(num_chains):
                for n_sample in tqdm(range(num_samples)):
                    expected_strain_signal = LorentzianModel.generate_strain(self.detector, self.frequencies, f0=fit.posterior['f0'].values[n_chain, n_sample], 
                                                                            gamma=fit.posterior['gamma'].values[n_chain, n_sample], 
                                                                            A=fit.posterior['A'].values[n_chain, n_sample], 
                                                                            phase=fit.posterior['phase'].values[n_chain, n_sample], t0=t0s )#+ fit.posterior['tjitter'].values[n_chain, n_sample])
                    
                    log_likelihood_postmerger = self.likelihood.log_likelihood(data, expected_strain_signal, noise_scale * self.detector.shot_noise_psd + self.detector.classical_noise_psd, self.frequencies)

                    cdf_dist = jnp.cumsum(jnp.exp(log_likelihood_postmerger - jnp.max(log_likelihood_postmerger)))/jnp.sum(jnp.exp(log_likelihood_postmerger - jnp.max(log_likelihood_postmerger)))

                    resampled_t0_vals = resampled_t0_vals.at[n_chain, n_sample].set(jnp.interp(np.random.uniform(), cdf_dist, t0s))

            fit.posterior['t0'] = (('chain','draw'),np.array(resampled_t0_vals))

        return fit

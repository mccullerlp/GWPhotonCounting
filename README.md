This is the codebase to undertake the analyses presented in "Photon counting readout for detection and inference of gravitational waves from neutron star merger remnants", as well as use the photon counting inference infrastructure for other calculations. 

Here is a summary what the folders included: 
1. `src/` contains the source code for the likelihoods, running the calculations, and examples on how to build the models. 
2. `examples/` contains some simple examples. This folder contains where Fig 1 is generated (though the code for this is contained in `src/GWPhotonCounting/plotting.py`). 
3. `projects/PM_EOS/` is where the scripts and notebooks live to produce all the results and figures. You will need to modify them for your computing resources. I ran these analyses on devices which use condor and so have the construction of dag files etc. IF you're on a SLURM architecture, then differnet setup is required. 
- `simple_comparison/` contains the code and analysis to generate Figs 2 to 5. 
- `snr_CI_distribution/` contains the dag file generation, code, and summary analysis to generate Fig 6. 
- `hierarchical_EOS/` contains the code to generate the hierarchical result in Fig 7. Specifically, most of the code is in the `marginalized_method/` subfolder.

Feel free to reach out with any questions, you have. If this code is used for any analysis, please cite the upcoming paper. 



# Orbit MCMC

***H. Fu, May 2025***

`orbit_mcmc` is a forward-modeling approach to determine orbital elements from
astrometric data of asteroids and planets. It is based on [skyfield](https://rhodesmill.org/skyfield/), [emcee](https://emcee.readthedocs.io/en/stable/), and [amoeba](./amoeba.py).

Two Jupyter notebooks are included in this repository:
- [simu_data.ipynb](./simu_data.ipynb) gives an example on how to
  generate simulated astrometric data for an asteroid and compare the
  predictions from skyfield with those from the [JPL Horizons](https://ssd.jpl.nasa.gov/horizons/). 
- [find_elements.ipynb](./find_elements.ipynb) loads the simulated
  observations, find the best-fit orbital elements with a downhill
  simplex method with simulated annealing (`amoeba_sa`), and runs a MCMC
  sampler (`emcee`) to characterize the posterior probabilility density 
  functions of the orbital elements. 

Some important functions (e.g., the data likelihood function and the
position predictor function) are defined in `orbit_mcmc_funs.py`. 

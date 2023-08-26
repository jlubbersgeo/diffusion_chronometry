# Modeling diffusion of trace elements in minerals

Welcome! This is a repository that contains many (useful?) things for [diffusion chronometry](https://www.nature.com/articles/s43017-020-0038-x) in magmatic systems. The main purpose of this threefold:

1. Provide some useful and transparent examples of how to set up diffusion models such that you can adapt the code for use in your own experiments
2. Provide "glass box" functions for some of the more complicated maths that can save you time in your own modeling
3. Keep a repository of models that have been used in various manuscripts over the course of my career such that results can be `relatively` easily reproduced.

## Current models

1. modeling Mg and Sr diffusion in plagioclase
2. general isotropic difusion modeling. This uses Sr in hornblende as an example, but in general can be applied to any cation-mineral pair that exhibits isotropic diffusion
3. Basic 3D diffusion using numerically generated data that represents Sr in sanidine.
   This is a repository for all things pertaining to the modelling of diffusive equilibration of trace elements in minerals. Currently there are models for:
4. A comparison for many of the various partitioning models in plagioclase and what it means for "equilibrium".

## `plag_diff.py`

`plag_diff.py` is a collection of functions that is used heavily in the Mg and Sr models for forward modeling diffusion in plagioclase. It contains functions for calculating partition coefficients in plagioclase, setting up model parameters (initial profiles, time grids, etc.), and a computationally efficient discretization implementation of Equation 7 (solution to Fick's 2<sup>nd</sup> Law for plag) from
[Costa et al., 2003](https://www.sciencedirect.com/science/article/pii/S0016703702013455). You can find out more about what's in there by:

```python
import plag_diff as plag
help(plag)
```

## List of Manuscripts

Below is a list of manuscripts that use `plag_diff.py` and the associated jupyter notebooks

- Jordan Lubbers, Adam J R Kent, Shanaka de Silva, Constraining magma storage conditions of the Toba magmatic system: a plagioclase and amphibole perspective, _in review at Contributions to Mineralogy and Petrology_

- [Jordan Lubbers, Adam J R Kent, Shanaka de Silva, Thermal Budgets of Magma Storage Constrained by Diffusion Chronometry: the Cerro Galán Ignimbrite, Journal of Petrology, Volume 63, Issue 7, July 2022, https://doi.org/10.1093/petrology/egac048](https://academic.oup.com/petrology/article/63/7/egac048/6601565)

## Dependencies

All the code in here uses the "typical" scipy stack. You should have no trouble running any of the code with the most up to date versions of the following:

- `pandas`: importing geochemical data, manipulating dataframes
- `NumPy`: the backbone of all numerical calculations
- `matplotlib`: visualization of data and model results
- `tqdm`: displaying inline progress bars while models run
- `SciPy`: statistical calculations, interpolation functions
- `mendeleev`: easy access to information on elements in the periodic table (ionic radii, charge, etc.)

## Misc. useful things

[Here](https://drive.google.com/file/d/1Tig0Ex6ZiVMGUX5Xusm2lfVL8LBtROBb/view?usp=sharing) is an example of how to code a fast and efficient diffusion equation that capitalizes on vectorized math operations from `NumPy`. It is about 70 times faster than using a scalar (i.e., double for-loop) approach. This is especially handy when your diffusion equation becomes more complicated. All the examples in this repository utilize this methodology.

For any questions or inquiries please reach out to Jordan Lubbers (jlubbers@usgs.gov)

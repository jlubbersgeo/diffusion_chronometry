# Modelling Diffusion of trace elements in minerals

This is a repository for all things pertaining to the modelling of diffusive equilibration of trace elements in minerals. Rather than build a bunch of fancy functions, the Jupyter notebooks are built "from scratch" so as to be transparent with as much of the building of the model as possible. Currently there are models for:

- trace elements (e.g., Mg and Sr) in plagioclase. This is built off the methodology initally described by [Costa et al., 2003](https://www.sciencedirect.com/science/article/pii/S0016703702013455)
- Isotropic diffusion modelling (i.e., Fick's 2<sup>nd</sup> Law). This uses the specific example of Sr diffusion in hornblende, however as there is no compositional dependence or crystallographic dependence on diffusion coefficient, the same logic can be used for any mineral - element pair that exhibits isotropic diffusion. 
- Basic 3D diffusion using numerically generated data that simulates Sr in sanidine. 

## Dependencies
- pandas: importing geochemical data, manipulating dataframes
- numpy: the backbone of all numerical calculations
- matplotlib: visualization of data and model results
- tqdm: displaying inline progress bars while models run
- scipy: statistical calculations, interpolation functions

Optionally, for quicker data visualization, seaborn is also used. 


## Useful things

[Here](https://drive.google.com/file/d/1Tig0Ex6ZiVMGUX5Xusm2lfVL8LBtROBb/view?usp=sharing) is an example of how to code a fast and efficient diffusion equation that capitalizes on vectorized math operations from Numpy. It is about 70 times faster than using a scalar (i.e., double for-loop) approach. This is especially handy when your diffusion equation becomes more complicated. All the examples in this repository utilize this methodology. 

More to come!

For any questions or inquiries reach out to Jordan Lubbers

# ADA-KRLS (adaptive dynamic adjustment kernel recursive least squares)

The adaptive dynamic adjustment kernel recursive least squares (ADA-KRLS) is a model proposed by Han et al. [1].

- [ADA-KRLS](https://github.com/kaikerochaalves/ADA-KRLS/blob/2d4e03bc21bb1e72d0ae77da8fa5efeb6f8b9a94/Model/ADA_KRLS.py) is the ADA-KRLS model.

- [GridSearch_AllDatasets](https://github.com/kaikerochaalves/ADA-KRLS/blob/2d4e03bc21bb1e72d0ae77da8fa5efeb6f8b9a94/GridSearch_AllDatasets.py) is the file to perform a grid search for all datasets and store the best hyper-parameters.

- [Runtime_AllDatasets](https://github.com/kaikerochaalves/ADA-KRLS/blob/2d4e03bc21bb1e72d0ae77da8fa5efeb6f8b9a94/Runtime_AllDatasets.py) perform 30 simulations for each dataset and compute the mean runtime and the standard deviation.

- [MackeyGlass](https://github.com/kaikerochaalves/ADA-KRLS/blob/2d4e03bc21bb1e72d0ae77da8fa5efeb6f8b9a94/MackeyGlass.py) is the script to prepare the Mackey-Glass time series, perform simulations, compute the results and plot the graphics. 

- [Nonlinear](https://github.com/kaikerochaalves/ADA-KRLS/blob/53b00c4bf70424c29396935ca6b9d32789e6f062/Nonlinear.py) is the script to prepare the nonlinear dynamic system identification time series, perform simulations, compute the results and plot the graphics.

- [LorenzAttractor](https://github.com/kaikerochaalves/ADA-KRLS/blob/2d4e03bc21bb1e72d0ae77da8fa5efeb6f8b9a94/LorenzAttractor.py) is the script to prepare the Lorenz Attractor time series, perform simulations, compute the results and plot the graphics. 

[1] M. Han, J. Ma, S. Kanae, Time series online prediction based on adaptive dynamic adjustment kernel recursive least squares algorithm, in: 2018 Ninth International Conference on Intelligent Control and Information Processing (ICICIP), IEEE, 2018, pp. 66–72. 
doi:https://doi.org/10.1001109/ICICIP.2018.8606696

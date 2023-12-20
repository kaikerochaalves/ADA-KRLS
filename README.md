# ADA-KRLS (adaptive dynamic adjustment kernel recursive least squares)

The adaptive dynamic adjustment kernel recursive least squares (ADA-KRLS) is a model proposed by Han et al. [1].

- [ADA-KRLS](https://github.com/kaikerochaalves/QKRLS/blob/6a1dcf72ecebd8473dc447df32b17ebb5b91e67a/Model/QKRLS.py) is the ADA-KRLS model.

- [GridSearch_AllDatasets](https://github.com/kaikerochaalves/QKRLS/blob/6a1dcf72ecebd8473dc447df32b17ebb5b91e67a/GridSearch_AllDatasets.py) is the file to perform a grid search for all datasets and store the best hyper-parameters.

- [Runtime_AllDatasets](https://github.com/kaikerochaalves/QKRLS/blob/6a1dcf72ecebd8473dc447df32b17ebb5b91e67a/Runtime_AllDatasets.py) perform 30 simulations for each dataset and compute the mean runtime and the standard deviation.

- [MackeyGlass](https://github.com/kaikerochaalves/QKRLS/blob/6a1dcf72ecebd8473dc447df32b17ebb5b91e67a/MackeyGlass.py) is the script to prepare the Mackey-Glass time series, perform simulations, compute the results and plot the graphics. 

- [Nonlinear](https://github.com/kaikerochaalves/QKRLS/blob/6a1dcf72ecebd8473dc447df32b17ebb5b91e67a/Nonlinear.py) is the script to prepare the nonlinear dynamic system identification time series, perform simulations, compute the results and plot the graphics.

- [LorenzAttractor](https://github.com/kaikerochaalves/QKRLS/blob/6a1dcf72ecebd8473dc447df32b17ebb5b91e67a/LorenzAttractor.py) is the script to prepare the Lorenz Attractor time series, perform simulations, compute the results and plot the graphics. 

[1] M. Han, J. Ma, S. Kanae, Time series online prediction based on adaptive dynamic adjustment kernel recursive least squares algorithm, in: 2018 Ninth International Conference on Intelligent Control and Information Processing (ICICIP), IEEE, 2018, pp. 66–72. 
doi:https://doi.org/10.1001109/ICICIP.2018.8606696

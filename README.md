# automlib
Python package with ML models with builtin Hyperparameter Optimization and easy to use API.

Hyperparameter optimization is one of the most important steps in any ML Pipeline. Often, is is accomplished using brute-force methods like Grid Search and Random Search. However, brute-force approaches have the following disadvantages:
  1. Its selection of the next hyperparameter settings to evaluate is not informed based on results it obtained from the previous settings.
  2. It takes a long amount of time to find a good solution
 
However, recently approaches using Bayesian Optimization, TPE have surfaced which are good solutions to the above mentioned problems. The approach in the present library uses Particle Swarm Optimization to optimize the ML models. Particle Swarm Optimization is an optimization procedure in which the trial solutions at each step are a function of the trial solutions at the previous step. Thus, the search progresses in an informed way - more search is performed in regions showing better settings.           

However, instead of providing a separate function/method for optimization, in automlib, it has been incorporated as part of the model wrapper. So there is no need to perform optimization separately - all we have to do is call the **fit** method and automlib optimizes the model automatically.

## Description
automlib currently contains the following model classes: 
* psoregressor
* psoclassifier

Both the classes use a lightgbm Gradient Boosting model(https://lightgbm.readthedocs.io/en/latest/) and a particle swarm optimization api from pyswarm(https://pythonhosted.org/pyswarm/) wrapped inside the psoregressor and psoclassifier class to produce the models. 
Following are some of the key parameters to set in the model(reasonable defaults are already set but one can experiment with different settings):
* population: Number of trial solutions to pursue at each iteration(default 30)
* maxiter: Maximum number of iterations to consider before terminating the search(default 100)
* minfunc: Minimum change in MSE(Mean Squared Error) of model before assuming convergence(default 1e-3). 

Increasing population and maxiter will yield better quality models but will take longer to produce results. Increasing the population parameter is more likely to give good results hovever, than increasing maxiter.  

Currently, the following parameters are being tuned:
* n_estimators
* max_depth
* max_features
* subsample
* learning_rate
* min_samples_leaf

## How to Use
Initialize and fit model:
```
import automlib

# Fit automlib model
model = automlib.psoregressor(population = 20,  maxiter = 30)
model.fit(X = X_train, y = y_train)
```
![Training Progress](/automlib_reg.PNG)

Predict with the model
```
y_predicted = model.predict(X_test)
```

Please view 'automlib documentation for regression model.html' for more details.

# automlib
Python package with ML models with builtin Hyperparameter Optimization and easy to use API.

Hyperparameter optimization is one of the most important steps in any ML Pipeline. Often, is is accomplished using brute-force methods like Grid Search and Random Search. However, brute-force approaches have the following disadvantages:
  1. Its selection of the next hyperparameter settings to evaluate is not informed based on results it obtained from the previous settings.
  2. It takes a long amount of time to find a good solution
 
However, recently approaches using Bayesian Optimization, TPE have surfaced which are good solutions to the above mentioned problems. The approach in the present library uses Particle Swarm Optimization to optimize the ML models. Particle Swarm Optimization is an optimization procedure in which the trial solutions at each step are a function of the trial solutions at the previous step. Thus, the search progresses in an informed way - more search is performed in regions showing better settings.           

However, instead of providing a separate function/method for optimization, in automlib, it has been incorporated as part of the model wrapper. So there is no need to perform optimization separately - all we have to do is call the **fit** method and automlib optimizes the model automatically.

### How to Use
Initialize model:
```
import automlib

# Fit automlib model
model = automlib.psoregressor(population = 20,  maxiter = 30)
model.fit(X = X_train, y = y_train)
```

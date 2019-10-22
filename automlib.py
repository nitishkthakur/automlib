import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, metrics, ensemble
from imblearn import over_sampling, under_sampling, combine
import pyswarm
import lightgbm as lgb


class psoregressor:
    def __init__(self, population = 30, omega=0.5, phip=0.5, phig=0.5, maxiter=100, minstep=1e-3,
    minfunc=1e-3, debug=True, params = {'n_estimators': [50, 2500], 'max_depth': [2, 10], 
                                                  'max_features': [.1, 1], 'subsample': [.1, 1], 
                                                  'learning_rate': [.01, .90], 'min_samples_leaf': [1, 400]}, cv = 5):
        
        self.pop = population
        self.omega = omega
        self.phip = phip
        self.phig = phig
        self.maxiter = maxiter
        self.minstep = minstep
        self.minfunc = minfunc
        self.debug = debug
        self.param_dictionary = params
        self.lb = [50, 2, .1, .1, .01, 1]
        self.ub = [2500, 10, 1, 1, .90, 400]
        self.hyperparameters = ['n_estimators', 'max_depth', 'max_features', 'subsample', 'learning_rate', 'min_samples_leaf']
        self.cv = cv
        
    def fit(self, X, y):
        #X_train_inner, X_test_inner, y_train_inner, y_test_inner = model_selection.train_test_split(X, y, test_size = 0.2, random_state = 40, shuffle = True)
        X = np.array(X)
        y = np.array(y)
        # Define Objective function
        def obj(x):
            params = {self.hyperparameters[0]: int(x[0]), self.hyperparameters[1]: int(x[1]), self.hyperparameters[2]: x[2], 
                     self.hyperparameters[3]: x[3], self.hyperparameters[4]: x[4], self.hyperparameters[5]: int(x[5])}
            
            
            kf = model_selection.KFold(n_splits=self.cv)
            kf.get_n_splits(X)
            
            rmse = []
            for train_index, test_index in kf.split(np.array(X)):
                X_train_inner, X_test_inner = np.array(X)[train_index], np.array(X)[test_index]
                y_train_inner, y_test_inner = np.array(y)[train_index], np.array(y)[test_index]
                
                model = lgb.LGBMRegressor(**params).fit(X_train_inner, y_train_inner)
                
                rmse.append(metrics.mean_squared_error(y_test_inner, model.predict(X_test_inner)))
            
            rmse = np.mean(rmse)
           
            return rmse
        
        # Perform Model Optimization
        xopt, fopt = pyswarm.pso(func = obj, lb = self.lb, ub = self.ub, swarmsize = self.pop, phip = self.phip, phig = self.phig,
                               omega = self.omega, maxiter =self.maxiter, minstep = self.minstep, minfunc = self.minfunc,
                                debug = self.debug)
        
        # Fit the best model
        hyperpara_optimized = {self.hyperparameters[0]: int(xopt[0]), self.hyperparameters[1]: int(xopt[1]), self.hyperparameters[2]: xopt[2], 
                     self.hyperparameters[3]: xopt[3], self.hyperparameters[4]: xopt[4], self.hyperparameters[5]: int(xopt[5])}
        
        self.fitted_model = lgb.LGBMRegressor(**hyperpara_optimized).fit(X, y)
        return self
    
    def predict(self, X):
        return self.fitted_model.predict(X)


#############################################################################################################################################
###### PSO based Classifier ########
## Lightgbm 

class psoclassifier:
    def __init__(self, params = {'n_estimators': [20, 2500], 'max_depth': [2, 10], 'min_data_in_leaf': [3, 200],
                                 'learning_rate': [0.01, 0.9], 'subsample': [0.1, 1], 'feature_fraction': [.01, 1],
                                'reg_lambda': [0.1, 5], 'num_leaves': [2, 700]},
                swarmsize = 25, omega=0.5, phip=0.5, phig=0.5, maxiter=100, minstep=1e-1,
    minfunc=1e-1, debug=True, cv = 5, top_n = 3, sample = 'oversample'):
        print('classifier imported')
        # Initialize hyperparameters of Optimizer
        self.swarmsize = swarmsize
        self.maxiter = maxiter
        self.omega = omega
        self.phip = phip
        self.phig = phig
        self.maxiter = maxiter
        self.minstep = minstep
        self.minfunc = minfunc
        self.debug = debug
        self.bounds = list(params.values())
        self.top_n = top_n
        self.fitted_status = False
        # Initialize model related parameters
        self.cv = cv
        self.params = params;
        self.param_names = list(params.keys())
        self.sample = sample
        # Print parameter details

        print('Parameters to tune and bounds: \n')
        print(pd.DataFrame(params, index = ['LB', 'UB']))
        # initialize model list
        self.model_list = []
        self.score = []
        self.upper_uncertainty = []
        self.lower_uncertainty = []
        
    def set_params(self, params_update):
        bounds_temp = list(params_update.values())
        param_names = list(params_update.keys())
        
        # Print parameter details
        print('Parameters to Update: ', param_names)
        print('New Parameter bounds: ', bounds_temp)
        
        for key in self.params:
            if key in param_names:
                print('Updating: ', key, ' from ', self.params[key], ' to ', params_update[key])
                self.params[key] = params_update[key]
        
    def get_params(self):
        return self.params.copy()
    
    def get_scores(self):
        return self.score.copy()
    
    def fit(self, X, y):
        # Get split indices
        kf = model_selection.StratifiedKFold(n_splits=self.cv, random_state = 0)
        kf.get_n_splits(X) 
        
        # initialize model list
        self.model_list = []
        self.score = []
        self.model_name = []
        print('\n\n Tuning Models \n\n')
        
        split_index = 1
        # Split into train test
        for train_index, test_index in kf.split(X,y):
            X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
            y_train, y_test = np.array(y).ravel()[train_index], np.array(y).ravel()[test_index]
            
            if self.sample == 'oversample':
                X_train, y_train = over_sampling.SMOTE(random_state = 12).fit_resample(X_train, y_train)
            if self.sample == 'undersample':
                X_train, y_train = under_sampling.EditedNearestNeighbours(random_state = 12).fit_resample(X_train, y_train)
            if self.sample == 'balance':
                X_train, y_train = combine.SMOTEENN(random_state = 12).fit_resample(X_train, y_train)
                
            # Define objective function
            def obj(x):
                params = {'n_estimators': int(x[0]), 'max_depth': int(x[1]), 'min_data_in_leaf': int(x[2]),
                                     'learning_rate': x[3], 'subsample': x[4], 'feature_fraction': x[5],
                         'reg_lambda': x[6], 'num_leaves': int(x[7])}

                # Fit required model:
                model_sel = lgb.LGBMClassifier(**params).fit(X_train, y_train, eval_set = [(X_test, y_test)],
                                                            eval_metric = 'multi_logloss', verbose = False)

                # Evaluate rmse
                score = -metrics.accuracy_score(y_test, model_sel.predict(X_test))

                return score
            
            if split_index == 1:
                ## Optimize
                
                pso = pyswarm.pso(func = obj, lb = [val[0] for val in self.bounds], ub = [val[1] for val in self.bounds], 
                                  swarmsize=self.swarmsize, omega=self.omega, phip=self.phip, phig=self.phig, 
                                  maxiter=self.maxiter,  minstep=self.minstep, minfunc=self.minfunc, debug=self.debug)

                # Get tuned hyperparameters
                x = pso[0]
                params_tuned = {'n_estimators': int(x[0]), 'max_depth': int(x[1]), 'min_data_in_leaf': int(x[2]),
                                         'learning_rate': x[3], 'subsample': x[4], 'feature_fraction': x[5],
                             'reg_lambda': x[6], 'num_leaves': int(x[7])}

                
            
            # Get Fitted, tuned model
            fitted_model = lgb.LGBMClassifier(**params_tuned).fit(X_train, y_train, eval_set = [(X_test, y_test)], 
                                                    eval_metric = 'multi_logloss', verbose = False)
            self.model_name.append('model' + str(split_index))
            
            # Calculate Score            
            score_temp = metrics.accuracy_score(y_test, fitted_model.predict(X_test))
            self.score.append(score_temp)
            print('Metric: ', score_temp, '\n\n')
            
            # Append model to final list of models
            self.model_list.append(fitted_model) 
            
            # Print fold errors
            print(self.score)
            split_index = split_index + 1
        model_rank = pd.DataFrame(data = list(zip(self.model_list, self.score)), 
                                  index = self.model_name, columns = ['model', 'score'])
        
        self.top_n_model_list = model_rank.sort_values('score')[:self.top_n].model.values.tolist()
        print('\n All Models trained: \n', model_rank['score'])
        #print('\n \n Models Selected by voting: \n \n', model_rank.sort_values('score')[:self.top_n]['score'])
        
        self.voting_classifier = ensemble.VotingClassifier(estimators = list(zip(self.model_name, self.model_list)), voting = 'soft').fit(X,y)
        
        self.fitted_status = True
        return self

    def predict(self, X):
        return self.voting_classifier.predict(X)

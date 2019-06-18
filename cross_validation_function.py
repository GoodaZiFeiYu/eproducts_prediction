'''
Author: Shikun Lin
date:06/10/2019

This file contains two functions that run k fold cross validation 
and gridsearch cross validation for all models

Please notice that the gridsearch cross validation will cost hours
to find the best hyperparameter for MLP. If you want to try this 
function, you can comment out the code for MLP

'''
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import timeit
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPRegressor


#This is a function which will run k fold validation 
# for  all modles. It will return the result of the 
#k fold with hyperparameters
#You can try k fold with best hyperparameters or without 
#the best hyperparameters by comment out the regressor initializer 
#Also print the time cost for all models
def k_fold_validation(x_train, x_test, y_train, y_test):
    
    all_accs = {}
    
    #validate ridge model 
    print("Start doing 10 folds validation for Ridge Regression...\n")
    ridge_time_start = timeit.default_timer()
 
    ridge_regressor = Ridge(alpha=1e-07,max_iter=300)
    #ridge_regressor = Ridge()
    accs = cross_val_score(estimator = ridge_regressor, 
        X=x_train,y=y_train,cv=10,n_jobs=-1)
    all_accs["ridge"] = accs
    
    ridge_time_stop = timeit.default_timer()
    
    print("Finshed for Ridge, time cost: {:.3f}s \n".format(ridge_time_stop-ridge_time_start))
    
    
    #validate lasso model
    print("Start doing 10 folds validation for Lasso Regression...\n")
    lasso_time_start = timeit.default_timer()
    lasso_regressor = Lasso(alpha=1e-06,max_iter=300)
    #lasso_regressor = Lasso()
    accs = cross_val_score(estimator = lasso_regressor, 
        X=x_train,y=y_train,cv=10,n_jobs=-1)
    all_accs["lasso"] = accs
    lasso_time_stop = timeit.default_timer()
    print("Finshed for Lasso, time cost: {:.3f}s \n".format(lasso_time_stop-lasso_time_start))
    
    
    # validate sgdregressor model
    print("Start doing 10 folds validation for SGD Regression...\n")
    sgd_time_start = timeit.default_timer()
    sgd_regressor = SGDRegressor(eta0= 2,power_t= 0.3 ,max_iter=300)
    #sgd_regressor = SGDRegressor()
    accs = cross_val_score(estimator = sgd_regressor, 
        X=x_train,y=y_train,cv=10,n_jobs=-1)
    all_accs["sgd"] = accs
    
    sgd_time_stop = timeit.default_timer()
    print("Finshed for SGD, time cost: {:.3f}s \n".format(sgd_time_stop-sgd_time_start))


    #validate multi layers perceptron model
    print("Start doing 10 folds validation for Multi Layer Perceptron...\n")
    
    mlp_time_start = timeit.default_timer()
    
    mlp_regressor = MLPRegressor(hidden_layer_sizes=(100, ),
        activation='tanh', solver='adam', 
        alpha=0.01, batch_size='auto', 
        learning_rate='constant', 
        learning_rate_init=0.001, power_t=0.5, 
        max_iter=200, shuffle=True, random_state=None, 
        tol=0.0001, verbose=False, warm_start=False,
        momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
        validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
        n_iter_no_change=10)
    
    
    #mlp_regressor = MLPRegressor()
    accs = cross_val_score(estimator = mlp_regressor, 
        X=x_train,y=y_train,cv=10,n_jobs=-1)
    all_accs["mlp"] = accs

    mlp_time_stop = timeit.default_timer()
    print("Finshed for MLP, time cost: {:.3f}s \n".format(mlp_time_stop-mlp_time_start))

    
 
    return all_accs

#This function can use grid search cross validation 
#to find out the best hyperparameters for all models
#Please notice that the gridsearch cross validation will cost hours
#to find the best hyperparameter for MLP. If you want to try this 
#function, you can comment out the code for MLP
#It will return the best accuracy and the best hyperparameters
def grid_search_sk(x_train, x_test, y_train, y_test):
    
    best_accs = {}
    
    best_paras = {}
    
    #run grid seach for ridge regression
    ridg_paras = [{'alpha': [1e-10,1e-7, 1e-6, 1e-3,1e-2, 0.1, 3, 10]
                   }]
    
    ridg_regressor = Ridge(alpha=1.0)
    
    grid_search_ridge = GridSearchCV(estimator = ridg_regressor,
                               param_grid = ridg_paras,
                               scoring = 'r2',
                               cv = 10,
                               n_jobs = -1
                               )
    
    grid_search_ridge.fit(x_train,y_train)
    best_acc_ridge = grid_search_ridge.best_score_
    best_para_ridge = grid_search_ridge.best_params_
    
    #save the result
    best_accs['ridge'] = best_acc_ridge
    
    best_paras['ridge'] = best_para_ridge
    
    lasso_paras = [{'alpha': [1e-10,1e-7, 1e-6, 1e-3,1e-2, 0.1, 3, 10]
                   }]
    

    #run grid seach for lasso regression
    lasso_regressor = Lasso(alpha=1.0,max_iter=300)
    
    grid_search_lasso = GridSearchCV(estimator = lasso_regressor,
                               param_grid = lasso_paras,
                               scoring = 'r2',
                               cv = 10,
                               n_jobs = -1
                               )
    
    grid_search_lasso.fit(x_train,y_train)
    best_acc_lasso = grid_search_lasso.best_score_
    best_para_lasso = grid_search_lasso.best_params_
    
    #save the result
    best_accs['lasso'] = best_acc_lasso
    
    best_paras['lasso'] = best_para_lasso
    
    #run the grid search for SGD regression
    sgd_regressor = SGDRegressor(eta0=1.0,power_t = 1.0)
    
    sgd_parameters = [
                {'eta0':[1e-3,1e-2, 0.1,0.2,0.3,0.4,0.5,0.8,1.0,2,3,5,7],
                 'power_t':[0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,2,3,5,7],
                    'max_iter':[300]
                    }
            ]
    
    grid_search_sgd = GridSearchCV(estimator = sgd_regressor,
                               param_grid = sgd_parameters,
                               scoring = 'r2',
                               cv = 10,
                               n_jobs = -1
                               )
    grid_search_sgd.fit(x_train,y_train)
    best_acc_sgd = grid_search_sgd.best_score_
    best_para_sgd = grid_search_sgd.best_params_
    
    #save the result
    best_accs['sgd'] = best_acc_sgd
    
    best_paras['sgd'] = best_para_sgd
    
    #run the grid search for Multi Layer Percptron.
    mlp_regressor = MLPRegressor(hidden_layer_sizes=(100, ),
        activation='relu', solver='adam', 
        alpha=0.0001, batch_size='auto', 
        learning_rate='constant', 
        learning_rate_init=0.001, power_t=0.5, 
        max_iter=300, shuffle=True, random_state=None, 
        tol=0.0001, verbose=False, warm_start=False,
        momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
        validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
        n_iter_no_change=10)
    
    mlp_paras = [{'hidden_layer_sizes':[(512,),(128,),(100,),(618,),(50,),(512,128,)],
                    'activation':['relu','tanh'],'solver':['sgd'],
                    'alpha':[1e-10,1e-7, 1e-6, 1e-3,1e-2, 0.1, 3, 10],
                    'max_iter':[10,50,100,200,300,500],
                    'learning_rate_init':[1e-6, 1e-3,1e-2, 0.1, 3, 10],
                    'momentum':[0.1,0.3,0.5,0.7,0.8,0.9],
                    'power_t':[0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,2,3,5,7]
            },
            {'hidden_layer_sizes':[(512,),(128,),(100,),(618,),(50,),(512,128,),
                        (128,100,),(618,128,),(50,10,)],
                    'activation':['relu','tanh'],'solver':['adam','lbfgs'],
                    'alpha':[1e-10,1e-7, 1e-6, 1e-3,1e-2, 0.1, 3, 10],
                    'max_iter':[10,50,100,200,300,500]
                    }
    
                    ]
            
            
    grid_search_mlp = GridSearchCV(estimator = mlp_regressor,
                               param_grid = mlp_paras,
                               scoring = 'r2',
                               cv = 10,
                               n_jobs = -1
                               )
    grid_search_mlp.fit(x_train,y_train)
    best_acc_mlp = grid_search_mlp.best_score_
    best_para_mlp = grid_search_mlp.best_params_
    
    #save the result
    best_accs['mlp'] = best_acc_mlp
    
    best_paras['mlp'] = best_para_mlp
    
    
    return best_accs,best_paras



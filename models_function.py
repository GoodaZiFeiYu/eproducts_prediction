'''
Author:Shikun Lin
Date: 06/10/2019

This file contains  all models and model function which 
can be uesed to predict the prices with best hyperparameters.

It also contains a plot function, which can plot the accuracy vs 
number of epoches

'''
# Importing the libraries
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

#multi layer perceptron function. It will return the accuracy and 
#predict prices.
def multi_layer_perceptron_sk(x_train,y_train,x_test,y_test,epochs):
    
    #create the regressor with best hyperparameters
    regressor = MLPRegressor(hidden_layer_sizes=(100, ),
        activation='tanh', solver='adam', 
        alpha=0.01, batch_size='auto', 
        learning_rate='constant', 
        learning_rate_init=0.001, power_t=0.5, 
        max_iter=epochs, shuffle=True, random_state=None, 
        tol=0.0001, verbose=False, warm_start=False,
        momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
        validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
        n_iter_no_change=10)

    #fit the data
    regressor.fit(x_train,y_train)
    
    #predict the prices
    y_pred = regressor.predict(x_test)
    #get the accuacy by r2
    acc = regressor.score(x_test,y_test)
    return y_pred,acc


#this is Laaso regression, it uses L1 regularization. 
# It will return the accuracy and predict prices.
def lasso_sk(x_train,y_train,x_test,y_test,epochs):
    
    #Create the lasso regressor with best hyperparameters
    regressor = Lasso(alpha=1e-06,max_iter=epochs)

    #fit the data
    regressor.fit(x_train,y_train)
    #predict prices
    y_pred = regressor.predict(x_test)
    #get the accuracy by r2
    acc = regressor.score(x_test,y_test)
    return y_pred,acc


#this is ridge regression. It uses L2 regularization.
#It will return the accuracy and predict prices.
def ridge_sk(x_train,y_train,x_test,y_test,epochs):
    #Create the ridge regressor with best hyperparameters
    regressor = Ridge(alpha=1e-07,max_iter=epochs)
    #fit the data
    regressor.fit(x_train,y_train)
    #pridict prices
    y_pred = regressor.predict(x_test)
    #get the accuracy by r2
    acc = regressor.score(x_test,y_test)
    return y_pred,acc

#this is SGD regression.
#It will return the accuracy and predict prices.
def sgdregressor_sk(x_train,y_train,x_test,y_test,epochs):

    #Create the SGD regressor with best hyperparameters
    regressor = SGDRegressor(eta0= 2,power_t= 0.3 ,max_iter=epochs)

    #fit the data
    regressor.fit(x_train,y_train)
    #predict the prices
    y_pred = regressor.predict(x_test)
    #get the accuarcy by r2
    acc = regressor.score(x_test,y_test)
    return y_pred,acc

#Plotting the accuracy vs epoches for specific models
def draw_acc_epoch(x_train, x_test, y_train, y_test,nmodle):
    acc_lst = []
    epoch_lst=[]
    
    #Plot the graph by name of the model
    if nmodle == 'sgd':
        for i in range(1,201):
            if i%5 ==0:
                preds,acc = sgdregressor_sk(x_train,y_train,x_test,y_test,i)
                acc_lst.append(acc)
                epoch_lst.append(i)
    
        plt.plot(epoch_lst,acc_lst)
        plt.title("{} accuracy for every epoch".format(nmodle))
        plt.ylabel("accuracy")
        plt.xlabel("epoch") 
        plt.show()
    if nmodle == 'mlp':
        for i in range(1,201):
            if i%5 ==0:
                preds,acc = multi_layer_perceptron_sk(x_train,y_train,x_test,y_test,i)
                acc_lst.append(acc)
                epoch_lst.append(i)
    
        plt.plot(epoch_lst,acc_lst)
        plt.title("{} accuracy for every epoch".format(nmodle))
        plt.ylabel("accuracy")
        plt.xlabel("epoch")  
        plt.show()
    if nmodle == 'ridge':
        for i in range(1,201):
            if i%5 ==0:
                preds,acc = ridge_sk(x_train,y_train,x_test,y_test,i)
                acc_lst.append(acc)
                epoch_lst.append(i)
    
        plt.plot(epoch_lst,acc_lst)
        plt.title("{} accuracy for every epoch".format(nmodle))
        plt.ylabel("accuracy")
        plt.xlabel("epoch")   
        plt.show()
    if nmodle == 'lasso':
        for i in range(1,201):
            if i%5 ==0:
                preds,acc = lasso_sk(x_train,y_train,x_test,y_test,i)
                acc_lst.append(acc)
                epoch_lst.append(i)
    
        plt.plot(epoch_lst,acc_lst)
        plt.title("{} accuracy for every epoch".format(nmodle))
        plt.ylabel("accuracy")
        plt.xlabel("epoch")   
        plt.show()
    else:
        print("Sorry, this {} model is not availabe. nmodle must in [ridge,lasso,sgd,mlp]\n".format(nmodle))

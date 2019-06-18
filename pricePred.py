'''
Author: Shikun Lin
Date: 06/10/2019
This is a project which use Ridge, Lasso, SGD, MLP to predict the 
prices of electronic prodcuts. And it will analysis the model, and 
find the best hyperparameter for all models. Also, it will give you
the best model from Ridge, Lasso, SGD, MLP!

'''


# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import Ridge
import timeit
from warnings import simplefilter,filterwarnings
from models_function import multi_layer_perceptron_sk, lasso_sk,ridge_sk,sgdregressor_sk,draw_acc_epoch
from cross_validation_function import k_fold_validation, grid_search_sk

#This data processing function, which can take the data which contains 
#about 15000 data of electronic products. 
def processing_data():
    ##read the csv file by using pandas
    dataset = pd.read_csv('DEPPData.csv')

    #ignore the warnning if the version of libs is not the newest
    simplefilter(action='ignore', category=DeprecationWarning)
    simplefilter(action='ignore', category=FutureWarning)

   
    
    #Upper the string of data. and add new column prices.amountAvg to data
    dataset = dataset[dataset['prices.currency'] == 'USD']
    dataset['prices.condition'] = dataset['prices.condition'].str.upper()
    dataset['categories'] = dataset['categories'].str.upper()
    dataset['brand'] = dataset['brand'].str.upper()
    dataset = dataset.fillna(0)
    dataset = dataset[dataset['prices.condition'] != 0 ]
    dataset = dataset[dataset['categories'] != 0 ]
    dataset = dataset[dataset['brand'] != 0 ]
    dataset['prices.amountAvg'] = (dataset['prices.amountMax']+dataset['prices.amountMin'])/2.0
    y = dataset.loc[:,['prices.amountAvg']].values
    

    #scaling the y vaule(average prices),You can try different scaler, it 
    # might effect the result.

    #sc = MinMaxScaler(feature_range=(0,1))
    sc = StandardScaler()
    y = sc.fit_transform(y)


    #drop some data that are not related to the prices prediction
    dataset = dataset.drop(['Unnamed: 26','Unnamed: 27','Unnamed: 28','Unnamed: 29','Unnamed: 30'],axis=1)
    
    dataset = dataset.drop(['id','prices.availability','prices.dateSeen','prices.merchant',
                            'prices.shipping','prices.sourceURLs','asins','dateAdded',
                            'dateUpdated','ean','imageURLs','keys','manufacturer',
                            'manufacturerNumber','name','primaryCategories','sourceURLs',
                            'upc','weight','prices.currency','prices.amountMax',
                            'prices.amountMin','prices.amountAvg'],axis=1)
    
    



    
    #Strating Select the best features by forward selection
    #Using onehot encode for those features.
    x = 0
    labels = dataset.axes[1].values.tolist()
    print(labels)
    selected_features = []
    accs = []

    #select the first feature
    accs.append(features_select(dataset,selected_features,labels,x,y))
    sf = accs[0]
    best_index = sf.index(max(sf))
    selected_features.append(labels[best_index])
    labels.remove(labels[best_index])
    #select second feature
    x = dataset.loc[:,[selected_features[0]]].values
    labelencoder_X = LabelEncoder()
    x[:, 0] = labelencoder_X.fit_transform(x[:, 0])
    onehotencoder = OneHotEncoder(categorical_features = [0])
    x = onehotencoder.fit_transform(x).toarray() 
    
    accs.append(features_select(dataset,selected_features,labels,x,y))
    sf = accs[1]
    if max(sf) > max(accs[0]):
        best_index = sf.index(max(sf))
    selected_features.append(labels[best_index])
    labels.remove(labels[best_index])
    
    #select third feature, but the accuracy is decreased 
    #when I added third featue into x
    x1 = dataset.loc[:,[selected_features[1]]].values
    labelencoder_X = LabelEncoder()
    x1[:, 0] = labelencoder_X.fit_transform(x1[:, 0])
    onehotencoder = OneHotEncoder(categorical_features = [0])
    x1 = onehotencoder.fit_transform(x1).toarray() 
    x =  np.c_[x,x1]
    accs.append(features_select(dataset,selected_features,labels,x,y))
    
    #At this point we find out "categories" and "brand" are two best features
    
    
    
    #Formatting the data and split it into training dataset 
    #and test_dataset

    x = dataset.loc[:,[selected_features[0]]].values
    labelencoder_X = LabelEncoder()
    x[:, 0] = labelencoder_X.fit_transform(x[:, 0])
    onehotencoder = OneHotEncoder(categories='auto')
    x = onehotencoder.fit_transform(x).toarray() 
    x1 = dataset.loc[:,[selected_features[1]]].values
    labelencoder_X = LabelEncoder()
    x1[:, 0] = labelencoder_X.fit_transform(x1[:, 0])
    onehotencoder = OneHotEncoder(categories='auto')
    x1 = onehotencoder.fit_transform(x1).toarray() 
    x =  np.c_[x,x1]

    #avoiding Dummy Variable Trap
    x = x[:,1:]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    
    
    return x_train, x_test, y_train, y_test,sc


#This is forward selecting function for features. It will return
#accuracy of the every feartures' combination
def features_select(dataset,selected_features,all_feature,x,y):
    
    accs = []
    if len(all_feature) == 4:
        for feature in all_feature:
       
            x_new= dataset.loc[:,[feature]].values
            labelencoder_X = LabelEncoder()
            x_new[:, 0] = labelencoder_X.fit_transform(x_new[:, 0])
            onehotencoder = OneHotEncoder(categorical_features = [0])
            x_new = onehotencoder.fit_transform(x_new).toarray()
            x_new = x_new[:,1:]
        
            x_train, x_test, y_train, y_test =train_test_split(x_new, y,test_size = 0.2, random_state = 0)
            
        
   
            acc = selected_features_helper(x_train,y_train,x_test,y_test)
            accs.append(acc)
    else:
        for feature in all_feature:
       
            x_new= dataset.loc[:,[feature]].values
            labelencoder_X = LabelEncoder()
            x_new[:, 0] = labelencoder_X.fit_transform(x_new[:, 0])
            onehotencoder = OneHotEncoder(categorical_features = [0])
            x_new = onehotencoder.fit_transform(x_new).toarray()
            x_new = np.c_[x,x_new]
            x_new= x_new[:,1:]
            
            x_train, x_test, y_train, y_test =train_test_split(x_new, y,test_size = 0.2, random_state = 0)
            
            acc = selected_features_helper(x_train,y_train,x_test,y_test)
            accs.append(acc)
    return accs

#This is the helper function for features_select function.
# It will return the accuracy for a specific dataset.
# It uses rigde regressor to predict   
def selected_features_helper(x_train,y_train,x_test,y_test):
    
    regressor = Ridge(alpha=1.0,max_iter=300)

    regressor.fit(x_train,y_train)
   
    acc = regressor.score(x_test,y_test)
    return acc




#This is the main function for the program. It will run data processing, 
#k fold cross validation, grid search validation and plot the graph. 
#If you run this program by command line. the program will be passus when the 
#plt graph pop up. So, you need to close the grap to continue the program
def main():
    
    #processing the data
    print("Program start Running ........\n")
    total_time_start = timeit.default_timer()
    
    print("Sart processing data and selecting the best features.....\n")
    data_loading_start = timeit.default_timer()
    x_train, x_test, y_train, y_test,sc = processing_data()
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    data_loading_stop = timeit.default_timer()
    
    print("Processing and selecting data finished! Time cost: {:.3f}s\n".format(
           data_loading_stop-data_loading_start))
    
    
    
    #start grid search. In order to save time, you can skip this becuase I already get the result
    print("Start choose best value of hyperparameters for all models by using grid search, it will take hours to compelete!\n")
    best_paras = "lasso: alpha = 1e-06, ridge: alpha = 1e-07, sgd: eta0 = 2, power_t=0.3, MLP:  activation=tanh,\
         solver=adam, alpha=0.01, learning_rate_init=0.001, max_iter=200 "

    #Please comment this line out, if you do not want to run the grid search cross vlaidation
    #If you skip, it will Not effect the program because all best hyperparameter 
    #have been selected and saved. However, if you want to try this function, you can comment
    #out the MLP secetion in grid_search_sk() function. This function is in cross_validation_function.py

    #best_accs,best_paras = grid_search_sk(x_train, x_test, y_train, y_test)
    
    print(best_paras)
    print("All hyperparameters are selected!!!\n")

    
    #Find the best model
    print("Starting select the best model by using 10 folds validation!!\n Please wait, it will take a while .....\n")
    validation_time_start = timeit.default_timer()
    models = ['ridge','lasso','sgd','mlp']
    accs = k_fold_validation(x_train, x_test, y_train, y_test)
    
    accs_all = []
    for mod in models:
        
        acc = accs[mod].mean()
        print("The average of k-fold validattion accuracy for {} is {} \n".format(mod,acc))
        accs_all.append(acc)
    
    best_mod = models[accs_all.index(max(accs_all))]

    validation_time_stop = timeit.default_timer()
    print("comparing finished, The Best Model by accuracy is {}！！！！！ time cost for all models: {:.3f}s\n".format(
        best_mod,validation_time_stop-validation_time_start))

    #calculate the accuracy for test dataset 
    print("Strat calculating the accuracy for test dataset")
    
    
    preds_1,acc_sgd = sgdregressor_sk(x_train,y_train,x_test,y_test,200)
    preds_2,acc_lasso = lasso_sk(x_train,y_train,x_test,y_test,200)
    preds_3,acc_ridge  = ridge_sk(x_train,y_train,x_test,y_test,200)
    preds_4, acc_mlp = multi_layer_perceptron_sk(x_train,y_train,x_test,y_test,200)

    print("test dataset, lasso: {}, ridge: {}, SGD: {}, MLP: {}".format(
                                acc_lasso,acc_ridge,acc_sgd,acc_mlp))



    #Plotting the graph
    #If you run this program by command line. the program will be pasus when the 
    #plt graph pop up. So, you need to close the grap to continue the program
    print("Start plot the accuracy with evry epoch for all models...\n")
    plot_time_start = timeit.default_timer()
    
    for mod in models:
        draw_acc_epoch(x_train, x_test, y_train, y_test,mod)
    
    plot_time_stop = timeit.default_timer()
    print("All models are plotted!!! time cost for plotting is {:.3f}s\n".format(plot_time_stop-plot_time_start))





    total_time_stop = timeit.default_timer()
    print('Total Running Time:{:.3f}s \n'.format(total_time_stop - total_time_start))

if __name__ == "__main__":
  main()





import pandas as pd
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from scipy.sparse import hstack,vstack
from sklearn.linear_model import LogisticRegression
import taskALoader as t_a
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC,SVC
from sklearn.utils import class_weight
from sklearn.metrics import classification_report,accuracy_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV,ShuffleSplit
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import LabelEncoder
from TfidfVectorizerWrapper import TfidfVectorizerWrapper
from ast import literal_eval
import pickle
from sklearn.ensemble import RandomForestClassifier
from statistics import mean,stdev
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score

def get_predefined_split(features_train, features_val, target_train, target_val):
    always_train = np.empty((target_train.shape[0]), dtype=np.int32)
    #print(target_train.shape[0])
    always_train[:] = -1
    always_validate = np.empty((target_val.shape[0]), dtype=np.int32)
    always_validate[:] = 0
    pred_split_indices = np.concatenate((always_train, always_validate))
    curr_target_train = np.concatenate((target_train, target_val))
    #curr_features_train = vstack((features_train, features_val))
    curr_features_train = np.concatenate((features_train, features_val))
   
    ps = PredefinedSplit(test_fold=pred_split_indices)

    return curr_features_train, curr_target_train, ps


def rf_param_selection(X_train,X_val,y_train,y_val):
    
    
    rfc=RandomForestClassifier(class_weight='balanced',random_state=42)
    

 
    
    
    
    X,y,ps = get_predefined_split (X_train,X_val,y_train,y_val)
    
  
    
    param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']}

                             
    
   

    grid_search = GridSearchCV(rfc, param_grid,cv=ps,scoring='f1_macro', refit=False)

    clf=grid_search.fit(X, y)
 
    
    grid_search.best_params_
    return grid_search.best_params_,clf

def runBest_RF(train,train_labels,test,test_labels,best_pars):
    
    
    n_estimators=best_pars['n_estimators']
    max_features=best_pars['max_features']
    max_depth=best_pars['max_depth']
    criterion=best_pars['criterion']
    
    
    
    clf= RandomForestClassifier( class_weight='balanced',n_estimators= n_estimators,max_features=max_features,max_depth=max_depth,
                                criterion=criterion,random_state=42)
    
    
    
    
    clf.fit(train,train_labels)
    predicted=clf.predict(test)
    print(classification_report(test_labels, predicted))
    print ("Accuracy: {}".format(accuracy_score(test_labels, predicted)))
    return clf



def logistic_param_selection(X_train,X_val,y_train,y_val):
    

    lr = LogisticRegression(class_weight='balanced',max_iter=20000)

    

    
    
    
    X,y,ps = get_predefined_split (X_train,X_val,y_train,y_val)
    
    #Logistic Regression hyperparameters
    #penalty = ['l1','l2']
    penalty = ['l2']
    Cs = [0.001, 0.01, 0.1, 1, 10]
    multi_class = ['multinomial','ovr']
    #multi_class = ['ovr']
    solver = ['newton-cg', 'lbfgs', 'sag']
    tol = [1e-3,1e-4,1e-5]
    #solver = ['liblinear']
    param_grid = {
                  "C": Cs,
                  "multi_class":multi_class,
                  "solver":solver,
                  "penalty":penalty,
                  "tol":tol
                 
                 }
        
                        
    
   

    grid_search = GridSearchCV(lr, param_grid,cv=ps,scoring='f1_macro', refit=False)

    clf=grid_search.fit(X, y)
 
    
    grid_search.best_params_
    return grid_search.best_params_,clf





def runBestLogistic(train,train_labels,test,test_labels,best_pars):
    
    
    c = best_pars['C']
    multi_class = best_pars['multi_class']
    penalty = best_pars['penalty']
    solver = best_pars['solver']
    tol = best_pars['tol']
    

    
    
    
    clf = LogisticRegression(C=c, multi_class= multi_class,penalty=penalty,solver=solver,
                             class_weight='balanced',max_iter=2000)
    
    
    
    
    clf.fit(train,train_labels)
    predicted=clf.predict(test)
    clf_report=classification_report(test_labels, predicted)
    #print ("Accuracy: {}".format(accuracy_score(test_labels, predicted)))
    f1=f1_score(test_labels, predicted, average='macro')  
    return clf,f1,clf_report











def gb_param_selection(X_train,X_val,y_train,y_val):
    
    
    rfc=GradientBoostingClassifier(random_state=42)
    

 
    
    
    
    X,y,ps = get_predefined_split (X_train,X_val,y_train,y_val)
    
    param_grid = {
    "loss":["deviance"],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[10]}

  
    
    
                             
    
   

    grid_search = GridSearchCV(rfc, param_grid,cv=ps,scoring='f1_macro', refit=False)

    clf=grid_search.fit(X, y)
 
    
    grid_search.best_params_
    return grid_search.best_params_,clf

def runBest_GB(train,train_labels,test,test_labels,best_pars):
    
    
    loss=best_pars["loss"]
    learning_rate=best_pars["learning_rate"]
    min_samples_split=best_pars["min_samples_split"]
    min_samples_leaf=best_pars["min_samples_leaf"]
    max_depth=best_pars["max_depth"]
    max_features=best_pars["max_features"]
    criterion=best_pars["criterion"]
    subsample=best_pars["subsample"]
    n_estimators=best_pars["n_estimators"]
    
    
    
    clf=GradientBoostingClassifier(n_estimators=n_estimators,subsample=subsample,criterion=criterion,loss=loss,learning_rate=learning_rate,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,max_depth=max_depth,max_features=max_features,random_state=42)
    
    
    
    
    clf.fit(train,train_labels)
    predicted=clf.predict(test)
    print(classification_report(test_labels, predicted))
    print ("Accuracy: {}".format(accuracy_score(test_labels, predicted)))
    return clf







def svc_param_selection(X_train,X_val,y_train,y_val):
    

    svc = LinearSVC(class_weight='balanced',max_iter=2000)
    

    
    
    X,y,ps = get_predefined_split (X_train,X_val,y_train,y_val)
    
    Cs = [0.001, 0.01, 0.1, 1, 10]
    Ls = ['l1','l2']
    multi_class = ['ovr','crammer_singer']
    #gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {
                  "C": Cs,
                  "multi_class":multi_class
                             
                 }
    
   
    
    grid_search = GridSearchCV(svc, param_grid,cv=ps,scoring='f1_macro', refit=False)
    
    clf=grid_search.fit(X, y)
 
    
    grid_search.best_params_
    return grid_search.best_params_,clf






def runBestSVM(train,train_labels,test,test_labels,best_pars):
    

    c = best_pars['C']
    multi_class = best_pars['multi_class']
 
   
    clf = LinearSVC(C=c,multi_class=multi_class,class_weight='balanced',max_iter=2000)
    
    
 

   
    clf.fit(train,train_labels)
    predicted=clf.predict(test)
    print(classification_report(test_labels, predicted))
    print ("Accuracy: {}".format(accuracy_score(test_labels, predicted)))
    return clf






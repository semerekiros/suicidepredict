#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
#class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
from langdetect import detect 


# In[3]:


df = pd.read_csv("data/task_A_dataset.csv").fillna(' ')
data = t_a.json_creator(df)
X,y = t_a.get_data(data,post_title_weight=True)



X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val  = train_test_split(X_train_all, y_train_all, test_size=0.25, random_state=1)


# In[ ]:





# In[ ]:





# In[4]:


t_a.display_class_dis(y_test)


# In[6]:


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



# In[7]:



X,y,ps = get_predefined_split (X_train,X_val,y_train,y_val)


# In[8]:


corpus_df = pd.read_csv("clpsych19_training_data/shared_task_posts.csv").fillna(' ')
p_title = corpus_df['post_title']
p_body = corpus_df['post_body']
corpus_text = pd.concat([p_title,p_body]) 


# In[9]:





# In[10]:





# In[11]:



class TfidfVectorizerWrapper(TfidfVectorizer):
   
   def fit(self,x,y=None, **fit_params):
       x = corpus_text
       return super(TfidfVectorizerWrapper, self).fit(x, y, **fit_params)
       
   def transform(self, x, y=None, **fit_params):
       #x = [content.split('\t')[0] for content in x]  # filtering the input
       return super(TfidfVectorizerWrapper, self).transform(x, y, **fit_params)       
       


# In[ ]:





# In[12]:


def svc_param_selection(X_train,X_val,y_train,y_val):
    

    
    linear_pipeline = Pipeline(
        [
            ('tfidf',
             TfidfVectorizerWrapper(decode_error='ignore', use_idf=True, ngram_range=(1, 2), lowercase=True,
                             stop_words='english',
                             analyzer='word', max_features=40000)),
            ('classifier', LinearSVC(class_weight='balanced',max_iter=40000))]
    )

    
    
    ngram_range=[(1, 2), (1, 3), (1, 4), (1, 5)]
    max_features=[100000, 200000, 300000, 400000, 500000]
    #max_features = [500000]
    min_df=[1, 2, 3]
    
    
    
    X,y,ps = get_predefined_split (X_train,X_val,y_train,y_val)
    
    Cs = [0.001, 0.01, 0.1, 1, 10]
    Ls = ['l1','l2']
    multi_class = ['ovr','crammer_singer']
    #gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {"tfidf__ngram_range": ngram_range,
                  "tfidf__use_idf": [True],
                  "tfidf__max_features": max_features,
                  "tfidf__min_df": min_df,
                  "tfidf__sublinear_tf": [True],
                  "classifier__C": Cs,
                  "classifier__multi_class":multi_class
                             
                 }
    
   
    
    grid_search = GridSearchCV(linear_pipeline, param_grid,cv=ps,scoring='f1_macro', refit=False)
    
    clf=grid_search.fit(X, y)
 
    
    grid_search.best_params_
    return grid_search.best_params_,clf


# In[ ]:


best_pars,model = svc_param_selection(X_train,X_val,y_train,y_val)


# In[36]:


def runBestSVM(train,train_labels,test,test_labels,best_pars):
    
    
    c = best_pars['classifier__C']
    multi_class = best_pars['classifier__multi_class']
    max_features = best_pars['tfidf__max_features']
    min_df = best_pars['tfidf__min_df']
    ngram_range=best_pars['tfidf__ngram_range']
    sublinear_tf = best_pars['tfidf__sublinear_tf']
    use_idf = best_pars['tfidf__use_idf']
    
    linear_pipeline = Pipeline(
        [
            ('tfidf',
             TfidfVectorizerWrapper(decode_error='ignore', 
                                    ngram_range=ngram_range, lowercase=True,stop_words='english',
                             analyzer='word', max_features=max_features,min_df=min_df,
                                    sublinear_tf=sublinear_tf,use_idf=use_idf)),
            
            ('classifier', LinearSVC(C=c,multi_class=multi_class,class_weight='balanced',max_iter=2000))]
    )
    
    
    
    
   
    linear_pipeline.fit(train,train_labels)
    predicted=linear_pipeline.predict(test)
    print(classification_report(test_labels, predicted))
    print ("Accuracy: {}".format(accuracy_score(test_labels, predicted)))
    return linear_pipeline


# In[38]:


model = runBestSVM(X_train,y_train,X_val,y_val,best_pars)   #Evaluation on dev set
runBestSVM(X_train,y_train,X_test,y_test,best_pars)         #Evaluation on test set


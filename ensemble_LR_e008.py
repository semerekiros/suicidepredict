#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
from textblob import TextBlob
from tqdm import tqdm_notebook as tqdm
import spacy
from modelsDenseFeatures import *


# In[2]:


nlp = spacy.load('en')



#df['new_col'] = df['text'].apply(lambda x: nlp(x))


# In[ ]:





# In[265]:


model_path = 'saved_models/'
filename = model_path+'605_Lb_logi_binary_clf_model.sav'
bin_clf_model = pickle.load(open(filename, 'rb'))




df = pd.read_csv("data/NewDataSet_with_Probs_297.csv").fillna(' ')   #Probs filled with leave one out
df_emotions = pd.read_csv("torchMojimaster/examples/emotions_dataset.csv").fillna(' ')
df_svm = pd.read_csv("data/NewDataSet_with_SVM_Probs_297.csv").fillna(' ')
df_emotion_probs = pd.read_csv("data/Emotion_NewDataSet_with_Probs.csv").fillna(' ')


# In[268]:


user_postss = list(df['posts'])
#df.columns


# In[272]:


all_posts = []
#all_sents = []
min_prob = []
max_prob = []
avg_prob = []
std_prob = []
post_count = []
sentiment_prob = []

tokens_len = []

for user in user_postss:
    one_user_posts = []
    one_user_sentiment = []
    one_user_tokens = ''
    
    for post_details in literal_eval(user):
        
        one_post =  str(post_details[3])+" "+str(post_details[4])
        '''
        prob = bin_clf_model.predict_proba([one_post])
        one_user_posts.append(prob[0][1].tolist())
        '''
        prob = bin_clf_model.decision_function([one_post])
        one_user_posts.append(prob[0].tolist())
        
        sentiment = TextBlob(one_post)
        one_user_sentiment.append(sentiment.sentiment[1])
        
        one_user_tokens = one_user_tokens + ' '+ str(post_details[3])+" "+str(post_details[4])
        
        
        #print(one_user_posts)
    all_posts.append(one_user_posts)
    sentiment_prob.append(max(one_user_sentiment))
    post_len = len(nlp.tokenizer(one_user_tokens))
    tokens_len.append(post_len)


# In[245]:






for usr_post in all_posts:
    if(len(usr_post)<2):
        std_risk = 0   
    else:
        std_risk = stdev(usr_post)
        
    max_risk = max(usr_post)
    avg_risk = mean(usr_post)
    min_risk = min(usr_post)
    
        
    max_prob.append(max_risk)
    avg_prob.append(avg_risk)
    std_prob.append(std_risk)
    min_prob.append(min_risk)
    post_count.append(len(usr_post))


# In[ ]:





# # Include SVM predictions

# In[246]:


svm_feat = df_svm.columns[5:]
svm_pred = df_svm[svm_feat]
df = pd.concat([df,svm_pred],axis=1)


# # Include the risk probs features from Annotated model

# In[247]:


df['max_risk'] = max_prob
df['avg_risk'] = avg_prob
df['std_risk'] = std_prob
#df['min_risk'] = min_prob
#df['sentiment_prob']= sentiment_prob
#df['post_count'] = post_count
#df['post_len'] = tokens_len


# # Include the emotion features 

# In[249]:


df = pd.concat([df,df_emotions],axis=1)


# In[ ]:





# In[251]:


features = df.columns[5:]  #Take all features
#features = df.columns[9:] #sentiment only
#features = df.columns[9:12]  #Take annotated features only

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
scaler = StandardScaler()
#scaler = MaxAbsScaler()
df[features] = scaler.fit_transform(df[features])

features


# In[253]:


train_df = df.iloc[0:297]
val_df = df.iloc[297:396]
test_df = df.iloc[396:]





X_train= train_df[features]
y_train = train_df['class']

X_val = val_df[features]
y_val = val_df['class']

X_test = test_df[features]
y_test = test_df['class']


# In[256]:





# In[257]:


import statistics as st

all_std = []
for col in features:
    vals = train_df[col]

 
    
    all_std.append(vals.values.std())


# In[259]:


print("Standard deviation along the different feature columns" , max(all_std)/float(min(all_std)))


# In[262]:


import warnings
warnings.filterwarnings('ignore')
best_pars,best_model = logistic_param_selection(X_train,X_val,y_train,y_val)


# In[233]:


model,f,clf_report = runBestLogistic(X_train,y_train,X_val,y_val,best_pars)
model1,f1,clf_report1=runBestLogistic(X_train,y_train,X_test,y_test,best_pars)
print("Dev set report\n")
print(clf_report)
print("Test set report\n")
print(clf_report1)


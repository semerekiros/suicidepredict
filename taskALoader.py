#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from scipy.sparse import hstack,vstack
from sklearn.linear_model import LogisticRegression


# In[11]:





# # Function to change dataframe to list of users with their labels and posts

# In[12]:


def json_creator(data_frame):
    
    data_in_json = []
    
    all_users = data_frame.groupby('user_id')
 
    for username, one_user in all_users:
        label = one_user.iloc[0]['raw_label']
        user_id = int(one_user.iloc[0]['user_id'])
        posts = []
        user = {}
        
        for index,row in one_user.iterrows():
            one_post = []
            one_post.append(row['post_id'])
            one_post.append(row['timestamp'])
            one_post.append(row['subreddit'])
            one_post.append(row['post_title'])
            one_post.append(row['post_body'])


            posts.append(one_post)

        user ['user_id'] = user_id
        user ['label'] = label
        user ['posts'] = posts
        data_in_json.append(user)
    return data_in_json


# In[ ]:





# # For each user concatenate all_posts+all_post_titles

# In[7]:


def concatenate_func(data,title_weight):

    x_data = []
    for user in data:
        label= user['label']
        all_posts =""
        for post in user['posts']:
            if(title_weight):
                all_posts=all_posts+" "+str(post[3])+" "+str(post[3])+" "+str(post[4])
            else:
                all_posts=all_posts+" "+str(post[3])+" "+str(post[4])
        x_data.append((all_posts,label))
        
    return x_data


# In[8]:



    




def get_data(data,post_title_weight):
    
    X = []
    y = []

    x_data = concatenate_func(data,post_title_weight)

    for user in x_data:
        X.append(user[0])
        y.append(user[1])

    le = preprocessing.LabelEncoder()
    y=le.fit_transform(y)
    print(le.inverse_transform([0,1,2,3]))
    return X,y


# # Function to Display class distribution

# In[9]:


def display_class_dis(y_labels):
    unique,counts=np.unique(y_labels, return_counts=True)
    percentage=dict(zip(unique,counts))

    print(percentage)
    s=sum(percentage.values())
    #l=len(percentage))  

    for k,v in percentage.items():
        print((v/float(s))*100)


    #objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
    objects = ['a','b','c','d']
    #y_pos = np.arange(len(objects))
    y_pos = np.arange(len(unique))
    #performance = [10,8,6,4,2,1]
    performance = counts
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Counts of samples')
    plt.title('Data distribution')

    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





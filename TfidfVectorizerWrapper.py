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




corpus_df = pd.read_csv("clpsych19_training_data/shared_task_posts.csv").fillna(' ')
p_title = corpus_df['post_title']
p_body = corpus_df['post_body']
corpus_text = pd.concat([p_title,p_body])




class TfidfVectorizerWrapper(TfidfVectorizer):
	def __init(self):
		self.corpus_df = pd.read_csv("clpsych19_training_data/shared_task_posts.csv").fillna(' ')
		self.p_title = corpus_df['post_title']
		self.p_body = corpus_df['post_body']
		self.corpus_text = pd.concat([p_title,p_body])




	def fit(self,x,y=None, **fit_params):
		x = self.corpus_text
		return super(TfidfVectorizerWrapper, self).fit(x, y, **fit_params)
		
	def transform(self, x, y=None, **fit_params):
		#x = [content.split('\t')[0] for content in x]  # filtering the input
		return super(TfidfVectorizerWrapper, self).transform(x, y, **fit_params)    

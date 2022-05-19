import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

review = pd.read_pickle('data/review_stop_lem.pickle')
data = pd.read_json('data/AMAZON_FASHION.json', lines=True)

X = review
y = data['overall']
del data #free some memory
X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size=0.25, random_state=42)

t = time.time()
vec = TfidfVectorizer(ngram_range=(1,2), max_df=0.5, max_features=50000)
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)
duration = time.time() - t
print("Time taken to extract features from data : %f seconds" % (duration))

t = time.time()
clf = MultinomialNB(alpha=0.05)
clf.fit(X_train_vec, y_train)
training_time = time.time() - t
print("train time: %0.3fs" % training_time)

y_pred = clf.predict(X_test_vec)
print("RMSE:", metrics.mean_squared_error(y_test, y_pred, squared=False))
print("MAE:", metrics.mean_absolute_error(y_test, y_pred))


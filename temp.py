# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np 
import pandas as pd 
dataset = pd.read_csv(r"C:\Users\Parvathi\Desktop\PROJECT\dataset (1).csv",error_bad_lines=False,encoding='latin-1')

import re #regular expression 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

data = []

for i in range(0,99989):

    #step1: replace regular expressions 
    sentiment = dataset["SentimentText"][i]
    sentiment = sentiment.lower()
    sentiment = sentiment.split()
    sentiment = [ps.stem(word) for word in sentiment if not word in set(stopwords.words('english'))]
    sentiment = ' '.join(sentiment)
    data.append(sentiment)

from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer(max_features = 1500)
x  = cv.fit_transform(data).toarray()
y = dataset.iloc[:,1:2].values
import pickle
pickle.dump(cv, open("cv.pkl", "wb"))

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

p = ["sad apl friend ", "miss new moon trailer", "omg alrea"]
cv2 = CountVectorizer(max_features = 9)
l = cv2.fit_transform(p).toarray()

'''
alrea  apl friend miss moon new omg sad trailer
0        1  1      0    0    0   0   1    0
0        0   0     1     1    1   0   0   1
1        0   0     0     0     0   1   0   0 '''

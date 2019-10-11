#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 10:38:25 2019

@author: Charlotte
"""

from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import nltk
import pandas as pd
import numpy as np 
import re

df_t = open('test.txt').readlines()

# lowercase all the words and remove url
stop_words = set(stopwords.words('english'))
def preprocess_tweet(tweet):
    words = tweet.split(' ')
    temp = []
    for word in words:
        s = word.lower()
        if "http" in s or s in stop_words:
            continue
        else:
            temp.append(s)
    return " ".join(temp)

def split_files(file):
    split = []
    for line in file:
        s = line.split('\t',2)
        split.append((preprocess_tweet(s[2]),s[1]))
    return split

def features(sentence):
    words = sentence.lower().split()
    return dict(('contains(%s)' % w, True) for w in words)


df_t = split_files(df_t)
df_t

input_paper = []
input_tweets = []
num_of_tweets = 0
for line in df_t:
    num_of_tweets = num_of_tweets+1
    input_id = line.split('\t',2)[0]
    s = line.split('\t',2)[2]
    input_paper.append((input_id,preprocess_tweet(s)))
    input_tweets.append(preprocess_tweet(s))


#using vader
input_vader_ans = []

analyzer = SentimentIntensityAnalyzer()
#sentences = input_tweets
for sentence in input_tweets:
    vs = analyzer.polarity_scores(sentence)
    if vs['compound'] >= 0.10: 
        input_vader_ans.append("positive") 
    elif vs['compound'] <= - 0.05:
        input_vader_ans.append("negative") 
    else : 
        input_vader_ans.append("neutral")

#using naive bayes
classify=nltk.classify.apply_features(features,df_t)
trainer=NaiveBayesClassifier.train(classify)

input_list = [features(t) for t in input_tweets]
input_nb_ans = [trainer.classify(t) for t in input_list]

ids = [tw[0] for tw in input_paper]

df = pd.DataFrame(list(zip(ids, input_nb_ans, input_vader_ans)), 
               columns =['ID', 'NB_ans', 'VD_ans'])
df
df.head(20)
df.to_csv('answer.csv')


df_test = pd.read_csv('test.txt', sep = '\t', header = None)
df_test.columns = ['id', 'label', 'text','1']
NBlabels = df.iloc[:, 1].values
VDlabels = df.iloc[:, 2].values
TRlabels = df_test['label']
Text = df_test['text']

NBlabels = list(NBlabels)
VDlabels = list(VDlabels)
TRlabels = list(TRlabels)
Text = list(Text)
df1 = pd.DataFrame(zip(NBlabels,TRlabels, Text),
                   columns = ['NBlabels','TRlabels','Text'])
print(df1.loc[[11]].values)

### F1 for Vader model
confusion_matrix(y_true=TRlabels, y_pred=VDlabels)
target_names = ['positive', 'neutral', 'negative']
score = metrics.precision_recall_fscore_support(y_true=TRlabels, y_pred=VDlabels)
print(classification_report(y_true=TRlabels, y_pred=VDlabels, target_names=target_names))

### F1 for Bayes model
confusion_matrix(y_true=TRlabels, y_pred=NBlabels)
target_names = ['positive', 'neutral', 'negative']
score = metrics.precision_recall_fscore_support(y_true=TRlabels, y_pred=NBlabels)
print(classification_report(y_true=TRlabels, y_pred=NBlabels, target_names=target_names))



# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 14:58:39 2020

@author: Faye
"""



import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import nltk
# read data
data = pd.read_csv("C:/Users/Faye/Downloads/REVIEWS.csv")
print(data)
print(data.head())
for i in data.columns:
    print(i)
print(len(data))



data_plot = data[["Review_Text","Review_Star_Rating"]].drop_duplicates()
data_plot_avg = data_plot.plot.hist()
print(plt.show())



pos_reviews = data.Review_Text
neg_reviews = data.Review_Text
print(type(pos_reviews))
pos_reviews_words = nltk.word_tokenize(pos_reviews[1])
print(pos_reviews_words)
print(type(pos_reviews[:5]))
print(len(pos_reviews))



pos_reviews_wordslist =[]
for i in range(87): #get error if put len+1 here, needed to switch from pos_reviews[1] to .iloc[1]
    pos_reviews_wordslist.append(nltk.word_tokenize(pos_reviews.iloc[i]))
    
print(pos_reviews_wordslist[:5])
print(len(pos_reviews_wordslist))
print(type(pos_reviews_wordslist))
neg_reviews_wordslist = [] #repeat tokenization for negative reviews
#for i in range(5):
for i in range(87): #get error if put len+1 here, needed to switch from pos_reviews[1] to .iloc[1]
    neg_reviews_wordslist.append(nltk.word_tokenize(neg_reviews.iloc[i]))
print(neg_reviews_wordslist[-5:])   


print(len(nltk.corpus.stopwords.words("english")))


nltk.corpus.stopwords.words("english")[:10]



useless_words = nltk.corpus.stopwords.words("english")
print(type(useless_words))



def build_bag_of_words_filtered(words):
    return {
        #word:1 for word in words
        word:1 
        for word in words \
        if not word in useless_words}

assert len(build_bag_of_words_filtered(["the"]))==0



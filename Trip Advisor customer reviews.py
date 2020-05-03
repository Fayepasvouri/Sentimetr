# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 05:04:29 2019

@author: Faye
"""

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
reviews=pd.read_csv("C:/Users/Faye/Downloads/REVIEWS.csv")
sia=SentimentIntensityAnalyzer()
print(sia)
reviews["neg"]=reviews["Review Text"].apply(lambda x:sia.polarity_scores(x)["neg"])
reviews["neu"]=reviews["Review Text"].apply(lambda x:sia.polarity_scores(x)["neu"])
reviews["pos"]=reviews["Review Text"].apply(lambda x:sia.polarity_scores(x)["pos"])
reviews["compound"]=reviews["Review Text"].apply(lambda x:sia.polarity_scores(x)["compound"])
print(reviews.head(90))

pos_review = [ j for i, j in enumerate(reviews['Review Text']) if reviews["compound"][i] > 0.2]
neu_review = [ j for i, j in enumerate(reviews['Review Text']) if 0.2>= reviews["compound"][i] >= -0.2]
neg_review = [ j for i, j in enumerate(reviews['Review Text']) if reviews["compound"][i] <-0.2]

print("Percentage of positive reviews: {}%".format(len(pos_review)*100/len(reviews["Review Text"])))

print("Percentage of neutral reviews: {}%".format(len(neu_review)*100/len(reviews["Review Text"])))

print("Percentage of negative reviews: {}%".format(len(neg_review)*100/len(reviews["Review Text"])))


from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from wordcloud import WordCloud, STOPWORDS

# get data directory (using getcwd() is needed to support running example in generated IPython notebook)
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

# Read the whole text.
text = open(path.join(d, 'C:/Users/Faye/Downloads/Reviews.txt')).read()

# read the mask image
# taken from
# http://www.stencilry.org/stencils/movies/alice%20in%20wonderland/255fk.jpg
alice_mask = np.array(Image.open(path.join(d, "alice_mask.png")))

stopwords = set(STOPWORDS)
stopwords.add("positive")

wc = WordCloud(background_color="white", max_words=50, mask=alice_mask,
               stopwords=stopwords, contour_width=10, contour_color='blue')

# generate word cloud
wc.generate(text)

# store to file
wc.to_file(path.join(d, "alice.png"))

# show
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.figure()
plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis("off")
plt.show()
import matplotlib.pyplot as plt
 
# create data
names='satisfied', 'neutral', 'disatisfied',
size=[98,2,0]
 
# Create a circle for the center of the plot
my_circle=plt.Circle( (0,0), 0.7, color='white')

plt.pie(size, labels=names, colors=['red','green','blue','skyblue'])
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
 
# Custom colors --> colors will cycle
plt.pie(size, labels=names, colors=['red','green'])
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
 
from palettable.colorbrewer.qualitative import Pastel1_7
plt.pie(size, labels=names, colors=Pastel1_7.hex_colors)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()


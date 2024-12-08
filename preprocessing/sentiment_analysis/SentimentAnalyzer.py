import csv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import pickle
import import_ipynb
import sys

#separate into its own file later maybe?
# 
class SentimentAnalyzer():
    #static
    sid = SentimentIntensityAnalyzer()

    def sentimentScore(comment):
        if (type(comment) != str):
            return 0
        
        sentiment_score = SentimentAnalyzer.sid.polarity_scores(comment)['compound']
        return sentiment_score
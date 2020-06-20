import csv
from text_normalization import *

class DataSrc:
    def __init__(self, path):
        super(DataSrc, self).__init__()
        self.tweet_intensity = self.load_raw_data(path)

    def load_raw_data(self, path):
        tweet_intensity = []
        f = open(path)
        tweet_reader = csv.reader(f, delimiter='\t')
        next(tweet_reader)
        for row in tweet_reader:
            tweet_intensity.append((row[1], row[3]))
        return tweet_intensity

    def get_normalized_tweets(self):
        tn = TextNormalization()
        normalized_tweets = []
        for ti in self.tweet_intensity:
            normalized_tweets.append(tn.lemma_sentence(ti[0]))
        return normalized_tweets

    def get_values(self):
        values = []
        for ti in self.tweet_intensity:
            values.append(ti[1][0:ti[1].find(':')])
        return values
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

np.random.seed(500)

# remove colums
train = pd.read_csv("2018-Valence-oc-En-train.txt", sep='\t')
dev = pd.read_csv("2018-Valence-oc-En-dev.txt", sep='\t')
train = train.drop(columns=['ID', 'Affect Dimension'])
dev = dev.drop(columns=['ID', 'Affect Dimension'])

# keep intensity num
train['Intensity Class']= [entry[0:entry.find(':')] for entry in train['Intensity Class']]
dev['Intensity Class']= [entry[0:entry.find(':')] for entry in dev['Intensity Class']]

# lowercase all text
train['Tweet'] = [entry.lower() for entry in train['Tweet']]
dev['Tweet'] = [entry.lower() for entry in dev['Tweet']]


# tokenize
train['Tweet'] = [word_tokenize(entry) for entry in train['Tweet']]
dev['Tweet'] = [word_tokenize(entry) for entry in dev['Tweet']]

# remove unimportant words & lemmatize based on the first tag
# mapping to wn
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index, entry in enumerate(train['Tweet']):
    final_words = []
    lemmatizer = WordNetLemmatizer()
    for word, tags in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            final_word = lemmatizer.lemmatize(word, tag_map[tags[0]])
            final_words.append(final_word)
    train.loc[index, 'Tweet_Final'] = str(final_words)

for index, entry in enumerate(dev['Tweet']):
    final_words = []
    lemmatizer = WordNetLemmatizer()
    for word, tags in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            final_word = lemmatizer.lemmatize(word, tag_map[tags[0]])
            final_words.append(final_word)
    dev.loc[index, 'Tweet_Final'] = str(final_words)

train_X = train['Tweet_Final']
train_Y = train['Intensity Class']
test_X = dev['Tweet_Final']
test_Y = dev['Intensity Class']

# numberize data
encoder = LabelEncoder()
train_Y = encoder.fit_transform(train_Y)
test_Y = encoder.fit_transform(test_Y)

# used to convert text to vector
vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(train_X.append(test_X))

# get vectors from text
train_X_vectors = vectorizer.transform(train_X)
test_X_vectors = vectorizer.transform(test_X)

# MultinomialNB
nb = MultinomialNB()
nb.fit(train_X_vectors, train_Y)
test_Y_predict = nb.predict(test_X_vectors)
print("--- MultinomialNB ---")
print(confusion_matrix(test_Y, test_Y_predict))
print(classification_report(test_Y, test_Y_predict))
print("===================")

# SVM
svc = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
svc.fit(train_X_vectors, train_Y)
test_Y_predict = svc.predict(test_X_vectors)
print("--- SVM ---")
print(confusion_matrix(test_Y, test_Y_predict))
print(classification_report(test_Y, test_Y_predict))
print("===================")

# KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_X_vectors, train_Y)
test_Y_predict = neigh.predict(test_X_vectors)
print("--- KNeighborsClassifier ---")
print(confusion_matrix(test_Y, test_Y_predict))
print(classification_report(test_Y, test_Y_predict))
print("===================")

# LogisticRegression
lr = LogisticRegression(random_state=0)
lr.fit(train_X_vectors, train_Y)
test_Y_predict = lr.predict(test_X_vectors)
print("--- LogisticRegression ---")
print(confusion_matrix(test_Y, test_Y_predict))
print(classification_report(test_Y, test_Y_predict))
print("===================")

# RandomForestClassifier
rf = RandomForestClassifier(max_depth=2, random_state=0)
rf.fit(train_X_vectors, train_Y)
test_Y_predict = rf.predict(test_X_vectors)
print("--- RandomForestClassifier ---")
print(confusion_matrix(test_Y, test_Y_predict))
print(classification_report(test_Y, test_Y_predict))
print("===================")

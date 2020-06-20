import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from data_src import *

class ClassicModels:
    def __init__(self, train_path, test_path):
        super(ClassicModels, self).__init__()
        self.train_data_src = DataSrc(train_path)
        self.test_data_src = DataSrc(test_path)
        # init vectorizer
        self.vectorizer = TfidfVectorizer()
        tws = self.train_data_src.get_normalized_tweets()
        tws.extend(self.test_data_src.get_normalized_tweets())
        self.vectorizer.fit(tws)

    def run_with_instance(self, instance):
        # train
        train_tweets = self.train_data_src.get_normalized_tweets()
        train_X = np.array(self.vectorizer.transform(train_tweets).toarray())
        train_y = np.array(self.train_data_src.get_values())
        instance.fit(train_X, train_y)
        # test
        test_tweets = self.test_data_src.get_normalized_tweets()
        test_X = np.array(self.vectorizer.transform(test_tweets).toarray())
        test_y = np.array(self.test_data_src.get_values())
        test_y_predict = instance.predict(test_X)
        # print result
        print(confusion_matrix(test_y, test_y_predict))
        print(classification_report(test_y, test_y_predict))

    def run(self):
        print('--- GaussianNB ---')
        gnb = GaussianNB()
        self.run_with_instance(gnb)
        print('====================')
        print('--- SVC ---')
        svc = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        self.run_with_instance(svc)
        print('====================')
        print('--- KNeighborsClassifier ---')
        neigh = KNeighborsClassifier(n_neighbors=3)
        self.run_with_instance(neigh)
        print('====================')
        print('--- LogisticRegression ---')
        lr = LogisticRegression(random_state=0)
        self.run_with_instance(lr)
        print('====================')
        print('--- RandomForestClassifier ---')
        rf = RandomForestClassifier(max_depth=2, random_state=0)
        self.run_with_instance(rf)
        print('====================')

train_path = 'dataset/Sentiment-Analysis-Task/2018-Valence-oc-En-train.txt'
test_path = 'dataset/Sentiment-Analysis-Task/2018-Valence-oc-En-dev.txt'
clm = ClassicModels(train_path, test_path)
clm.run()


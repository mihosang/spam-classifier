#!/usr/bin/env python
import os

import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from scipy import sparse

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import BaseEstimator

#Word embedding
import gensim
from gensim.models import word2vec

class NBFeaturer(BaseEstimator):
    def __init__(self, alpha):
        self.alpha = alpha

    def preprocess_x(self, x, r):
        return x.multiply(r)

    def pr(self, x, y_i, y):
        p = x[y == y_i].sum(0)
        return (p + self.alpha) / ((y == y_i).sum() + self.alpha)

    def fit(self, x, y=None):
        self._r = sparse.csr_matrix(np.log(self.pr(x, 1, y) / self.pr(x, 0, y)))
        return self

    def transform(self, x):
        x_nb = self.preprocess_x(x, self._r)
        return x_nb


class Classifier():

    def __init__(self, model):
        self.model = model

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test, y_test):
        self.y_test = y_test
        self.prediction = self.model.predict(x_test)
        self.score = accuracy_score(y_test, self.prediction)

    def get_score(self):
        return self.score

    def print_report(self):
        print(" 1. Classification Report")
        print(classification_report(self.y_test, self.prediction, target_names=["Ham", "Spam"]))
        print(" 2. Confusion Report")
        conf_mat = confusion_matrix(self.y_test, self.prediction)
        print(conf_mat)


def import_spam_data(path):
    data = pd.read_csv(path, encoding='latin-1')
    # Drop column and name change
    data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
    data = data.rename(columns={"v1": "label", "v2": "text"})
    # convert label to a numerical variable
    data['label_num'] = data.label.map({'ham': 0, 'spam': 1})

    # print_data(data)
    return data

def import_phishing_data(path):
    data = pd.read_json(path)
    data = data.drop(["audioId"], axis=1)
    data['label'] = data.phishing.map({True:'spam', False: 'ham'})
    print(data)

def print_data(data):
    print(data.head())
    print(data.tail())
    print(data.label.value_counts())
    print(data.label.map({'ham': 0, 'spam': 1}))
    print(data.head())


def get_document_term_matrix(X_train, X_test):
    # vect = CountVectorizer(stop_words=stopwords.words('english'))
    vect = CountVectorizer()
    # vect = TfidfVectorizer(stop_words=stopwords.words('english'))
    # vect.fit(X_train)
    dtm = vect.fit_transform(X_train)
    print(
        'fit_transform, (sentence {}, feature {})'.format(dtm.shape[0], dtm.shape[1])
    )
    print(dtm.toarray())
    print(vect.get_feature_names())
    print(vect.get_stop_words())
    print(vect.get_params())
    X_train_df = vect.transform(X_train)
    X_test_df = vect.transform(X_test)

    return X_train_df, X_test_df

def spam_model(model_file, training_contents, training_category):

    # vect = TfidfVectorizer(use_idf=True)
    vect = CountVectorizer()
    mnb = MultinomialNB()
    nb = NBFeaturer(1)

    pip=Pipeline([('Vectorizer', vect), ('mnb', mnb)])
    clf = pip.fit(training_contents, training_category)

    joblib.dump(clf, model_file)

if __name__ == "__main__":
    # with open("./data/spam.csv", encoding='latin-1') as r:
    #     text = r.readlines()
    #     token = [s.split() for s in text]
    # print(token)
    import_phishing_data("./mldataset/phishing/abnormal/fishing-2018-10-17-140559.json")

    print("Loading data")
    # STEP 1. Import & Prepare dataset
    data = import_spam_data("./data/spam.csv")
    # STEP 2. Split into Train & Test sets
    X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.2, random_state=10)
    # STEP 3. Text Transformation
    X_train_df, X_test_df = get_document_term_matrix(X_train, X_test)
    # STEP 4. Classifiers
    prediction = dict()

    from konlpy.tag import Twitter

    test_text = u'단독입찰보다 복수입찰의 경우'
    nlp = Twitter()
    print(nlp.morphs(test_text))
    print(nlp.nouns(test_text))
    print(nlp.phrases(test_text))
    print(nlp.pos(test_text))

    # exit(0)
    # STEP 4.1. MultinomialNB Classifier
    nb_classifier = Classifier(MultinomialNB())
    nb_classifier.train(X_train_df, y_train)
    nb_classifier.predict(X_test_df, y_test)
    prediction["Multinomial"] = nb_classifier.get_score()

    # model_filename = 'spam_model_eng_mnb.pkl'
    # spam_model(model_filename, X_train, y_train)
    #
    # clf = joblib.load(model_filename)
    #
    # print(clf.predict_proba(y_train))
    # print(prediction)

    # exit(0)
    # STEP 4.2. Logistic Regression
    lr_classifier = Classifier(LogisticRegression())
    lr_classifier.train(X_train_df, y_train)
    lr_classifier.predict(X_test_df, y_test)
    prediction["LogisticRegression"] = lr_classifier.get_score()

    # STEP 4.3. k-NN Classifier
    knn_classifier = Classifier(KNeighborsClassifier(n_neighbors=5))
    knn_classifier.train(X_train_df, y_train)
    knn_classifier.predict(X_test_df, y_test)
    prediction["k-NN"] = knn_classifier.get_score()

    # STEP 4.4. Random Forest Classifier
    rf_classifier = Classifier(RandomForestClassifier())
    rf_classifier.train(X_train_df, y_train)
    rf_classifier.predict(X_test_df, y_test)
    prediction["RandomForest"] = rf_classifier.get_score()

    # STEP 4.4. Adaboost
    ab_classifier = Classifier(AdaBoostClassifier())
    ab_classifier.train(X_train_df, y_train)
    ab_classifier.predict(X_test_df, y_test)
    prediction["AdaBoost"] = ab_classifier.get_score()

    # STEP 4.5. SVN
    svm_classifier = Classifier(SVC())
    svm_classifier.train(X_train_df, y_train)
    svm_classifier.predict(X_test_df, y_test)
    prediction["SVM"] = svm_classifier.get_score()
    # nb_classifier.print_report()
    print(prediction)


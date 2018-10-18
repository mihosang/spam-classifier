#!/usr/bin/env python
from pprint import pprint
import pandas as pd

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.externals import joblib

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

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

    def save_model(self, model_file):
        joblib.dump(self.model, model_file)

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
    # print_data(data)
    return data

def print_data(data):
    print(data.head())
    print(data.tail())
    print(data.label.value_counts())
    print(data.label.map({'ham': 0, 'spam': 1}))
    print(data.head())

def tokenizer_korean(text):
    pass

def get_document_term_matrix_korean(X_train, X_test):
    from konlpy.tag import Okt
    t = Okt()

    stopwords_korean = set()
    with open("./data/stopwords-ko/ranksnl-korean.txt","r") as f:
        temp = f.readlines()
        for l in temp:
            temp_morphs = t.morphs(l.strip('\r\n'))
            for item in temp_morphs:
                stopwords_korean.add(item)
    with open("./data/stopwords-ko/ranksnl-korean.txt","r") as f:
        temp = f.readlines()
        temps = [l.strip('\r\n') for l in temp]

    stopwords_korean = list(stopwords_korean)
    print(temps)
    print(stopwords_korean)

    vect = CountVectorizer(stop_words=stopwords_korean, ngram_range=(1, 1))
    vect.fit(X_train)
    print(vect.vocabulary_)

    X_train_df = vect.transform(X_train)
    X_test_df = vect.transform(X_test)

    return X_train_df, X_test_df

def get_document_term_matrix(X_train, X_test):

    vect = CountVectorizer(stop_words=stopwords.words('english'))
    vect.fit(X_train)

    X_train_df = vect.transform(X_train)
    X_test_df = vect.transform(X_test)

    return X_train_df, X_test_df

if __name__ == "__main__":

    print("Loading data")
    # STEP 1. Import & Prepare dataset
    data = import_spam_data("./data/spam.csv")
    # data = import_phishing_data("./mldataset/phishing/abnormal/fishing-2018-10-17-184447.json")

    # STEP 2. Split into Train & Test sets
    X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.2, random_state=10)
    # STEP 3. Text Transformation
    X_train_df, X_test_df = get_document_term_matrix(X_train, X_test)
    # X_train_df, X_test_df = get_document_term_matrix_korean(X_train, X_test)
    # STEP 4. Classifiers

    all_models = [
        ("Multinomial", MultinomialNB()),
        ("LogisticRegression", LogisticRegression(solver='liblinear')),
        ("k-NN", KNeighborsClassifier(n_neighbors=5)),
        ("RandomForest",RandomForestClassifier(n_estimators=100)),
        ("Adaboost",AdaBoostClassifier()),
        ("C-SVC",SVC(gamma='auto'))

    ]
    
    scores = dict()
    for model in all_models:
        classifier = Classifier(model[1])
        classifier.train(X_train_df, y_train)
        classifier.predict(X_test_df, y_test)
        classifier.save_model("./data/%s-model.pkl" % model[0])
        scores[model[0]] = classifier.get_score()

    pprint(scores)
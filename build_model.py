#!/usr/bin/env python
import json
from pprint import pprint

import numpy as np
import pandas as pd
from gensim.models import word2vec
from konlpy.tag import Okt
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from classifier import Classifier


def make_wordvector(sentences):
    import multiprocessing
    cores = multiprocessing.cpu_count()

    num_features = 300  # 문자 벡터 차원 수
    min_word_count = 2  # 최소 단어 수
    num_workers = cores  # 병렬 처리 스레드 수
    context = 10  # 주변 단어 몇개까지

    model = word2vec.Word2Vec(sentences,
                              workers=num_workers,
                              size=num_features,
                              min_count=min_word_count,
                              window=context,
                              sg=1)
    return model


# tokenizer : 문장에서 색인어 추출을 위해 명사,동사,알파벳,숫자 정도의 단어만 뽑아서 normalization, stemming 처리하도록 함
def tokenizer(raw, pos={"Noun", "Verb"}, stopword={}):
    okt = Okt(max_heap_size=4096)
    return [
        word for word, tag in okt.pos(
            raw,
            norm=True,  # normalize 그랰ㅋㅋ -> 그래ㅋㅋ
            stem=True  # stemming 바뀌나->바뀌다
        )
        if len(word) > 1 and tag in pos and word not in stopword
    ]


def get_stopword():
    okt = Okt(max_heap_size=4096)
    filelist = [
        "./data/stopwords-ko/geonetwork-kor.txt",
        "./data/stopwords-ko/gh-stopwords-json-ko.txt",
        "./data/stopwords-ko/ranksnl-korean.txt"
    ]
    stopwords_korean = {
        "여보세요","에스","케이","음", "네네", "상담원", "span", "color", "blue", "weight", "font","고객님","감사합니다","알겠습니다",
        "고맙습니다","상담사","되세요","지금", "제가", "입니다", "번호가", "bold","style","하다", "고객", "되다", "번호", "감사하다","이다","요금","알다"
    }
    for filename in filelist:
        with open(filename, "r") as f:
            temp = f.readlines()
            for l in temp:
                stopwords_korean.update(okt.morphs(l.strip('\r\n')))
    return stopwords_korean


def import_phishing_nouns(paths):
    _stopword = get_stopword()
    dataset = None
    for path in paths:
        with open(path, encoding='utf-8') as f:
            lines = json.load(f)
            sentences = []
            for raw in lines:
                item = {}
                item['label'] = 1 if raw['phishing'] is True else 0
                item['text'] = ' '.join(tokenizer(raw['text'], stopword=_stopword))
                sentences.append(item)

            data = pd.DataFrame(sentences)

        print("len(data) = %s" % len(data))
        if dataset is None:
            dataset = data
        else:
            dataset = pd.concat([dataset, data], sort=False)
    return dataset


def import_phishing_data(paths):
    if len(paths) == 0:
        print("ERROR: Empty input file paths")
        return pd.DataFrame()
    dataset = None
    for path in paths:
        data = pd.read_json(path, encoding='utf-8')
        data['label'] = data.phishing.map({True: 1, False: 0})
        data = data.drop(["audioId"], axis=1)
        # data = data.take(np.random.permutation(len(data))[:500])
        print("len(data) = %s" % len(data))
        if dataset is None:
            dataset = data
        else:
            dataset = pd.concat([dataset, data], sort=False)
    return dataset


def print_data(data):
    print(data.head())
    print(data.tail())
    print(data.label.value_counts())
    print(data.label.map({'normal': 0, 'phishing': 1}))
    print(data.head())


def get_document_term_matrix_korean(X_train, X_test):
    # vect = CountVectorizer(tokenizer=tokenizer,lowercase=False, ngram_range=(1,2))
    vect = TfidfVectorizer(lowercase=False,tokenizer=tokenizer)
    vect.fit(X_train)

    joblib.dump(vect, 'data/model/vectorizer.joblib')

    X_train_df = vect.transform(X_train)
    X_test_df = vect.transform(X_test)

    X = vect.fit_transform(X_train)
    print('sentences {} feature {}'.format(X.shape[0], X.shape[1]))

    feature = vect.get_feature_names()
    # print(feature)
    return X_train_df, X_test_df


if __name__ == "__main__":

    print("Loading data")
    # STEP 1. Import & Prepare dataset
    data = import_phishing_data([
        "./mldataset/phishing/abnormal/phishing-2018-10-18-161054.json",
        "./mldataset/phishing/abnormal/phishing-meta-2018-10-23-115644.json",
        # "./mldataset/phishing/normal/logo.csv-AsBoolean.json",
        "./mldataset/phishing/normal/txt_20180711.final.json"
        # "./mldataset/phishing/normal/normal-2018-10-18-210902.json"
    ])

    # STEP 2. Split into Train & Test sets
    X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.3, random_state=10)
    # STEP 3. Text Transformation
    X_train_df, X_test_df = get_document_term_matrix_korean(X_train, X_test)
    # STEP 4. Classifiers
    all_models = [
        ("Multinomial", MultinomialNB()),
        # ("LogisticRegression", LogisticRegression(solver='liblinear')),
        # ("RandomForest", RandomForestClassifier(n_estimators=100)),
        # ("Adaboost", AdaBoostClassifier()),
        ("C-SVC", SVC(kernel='linear', probability=True))
    ]
    scores = []
    for model in all_models:
        classifier = Classifier(model[1])
        classifier.train(X_train_df, y_train)
        classifier.predict(X_test_df, y_test)
        classifier.save_model("./data/%s-model.pkl" % model[0])
        classifier.print_report()
        scores.append({"model": model[0], "score": float(classifier.get_score())})

    sorted_scores = sorted(scores, key=lambda k: k['score'], reverse=True)
    pprint(sorted_scores)

    # Random Forest Importance Matrix
    forest = RandomForestClassifier(n_estimators=100)
    forest.fit(X_train_df, y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    # for f in range(X_train_df.shape[1]):
    vect = joblib.load('vectorizer.joblib')
    for f in range(10):
        print("%d. feature %d %s (%f)" % (f + 1, indices[f], vect.get_feature_names()[indices[f]], importances[indices[f]]))


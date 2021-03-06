from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class Classifier():

    def __init__(self, model):
        self.model = model

    def train(self, x_train, y_train):
        return self.model.fit(x_train, y_train)

    def predict(self, x_test, y_test):
        self.y_test = y_test
        self.prediction = self.model.predict(x_test)
        self.score = accuracy_score(y_test, self.prediction)

    def get_score(self):
        return self.score

    def print_report(self):
        print("Report for classifier %s" % self.model)
        print(" 1. Classification Report")
        print(classification_report(self.y_test, self.prediction, target_names=["Ham", "Spam"]))
        print(" 2. Confusion Report")
        print(confusion_matrix(self.y_test, self.prediction))

    def save_model(self, model_file):
        self.model_file = model_file
        joblib.dump(self.model, model_file)
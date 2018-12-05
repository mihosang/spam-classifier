import pickle

from flask import Flask, jsonify
from flask_restful import Resource, Api, request, reqparse
from keras.engine.saving import model_from_json
from sklearn.externals import joblib
from build_model import tokenizer
from keras.preprocessing import sequence

max_words = 1000
max_len = 150

app = Flask(__name__)
api = Api(app)

class PredictionResource(Resource):
    def __init__(self):
        with open('data/model/lstm_model.json', 'r') as handle:
            self.json_file = handle.read()
        self.model = model_from_json(self.json_file)
        self.model.load_weights('data/model/lstm_model.h5')
        with open('data/model/tokenizer.pickle', 'rb') as handle:
            self.tok = pickle.load(handle)
        self.svc_model = joblib.load('data/model/C-SVC-model.pkl')
        self.vect = joblib.load('data/model/vectorizer.joblib')

    def svc_predict(self, text):
        x_test = self.vect.transform(text)
        predict = self.svc_model.predict(x_test)
        predict_proba = self.svc_model.predict_proba(x_test)
        l_predict_proba = []
        for i in range(0, len(text)):
            l_predict_proba.append(predict_proba[i][1])
        predict = ["phishing" if x == 1 else "normal" for x in predict]
        l_predict_proba = [str(x) for x in l_predict_proba]
        return predict, l_predict_proba

    def lstm_predict(self, text):

        sequence_matrix = sequence.pad_sequences(self.tok.texts_to_sequences(text), maxlen=max_len)
        prediction = self.model.predict(sequence_matrix)
        l_predict_proba = []
        for i in range(0, len(text)):
            l_predict_proba.append(prediction[i][0])
        predict = ["phishing" if x > 0.5 else "normal" for x in l_predict_proba]
        l_predict_proba = [str(x) for x in l_predict_proba]

        return predict, l_predict_proba

    def post(self):
        try:
            body = request.get_json()
            text = body['text']
            if 'model' in body:
                model = body['model']
            else:
                model = 'all'

            if type(text) is str:
                l_text = [text]
            elif type(text) is list:
                l_text = text
            else:
                return jsonify({"error": "text field is required as string or list of string"})

            if model == 'svc':
                svc_predict, svc_predict_proba = self.svc_predict(l_text)
                result = jsonify({"result": svc_predict, "probability": svc_predict_proba})
            elif model == 'lstm':
                lstm_predict, lstm_predict_proba = self.lstm_predict(l_text)
                result = jsonify({"result": lstm_predict, "probability": lstm_predict_proba})
            else:
                svc_predict, svc_predict_proba = self.svc_predict(l_text)
                lstm_predict, lstm_predict_proba = self.lstm_predict(l_text)

                result = jsonify(
                    {
                        "svc": {"result": svc_predict, "probability": svc_predict_proba},
                        "lstm": {"result": lstm_predict, "probability": lstm_predict_proba}
                     }
                )

            from keras import backend as K
            K.clear_session()

            return result

        except Exception as err:
            print(err)
            return jsonify({"error": str(err)})

class TrainingResource(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('text')
            parser.add_argument('label')
            args = parser.parse_args()

            text, label = args['text'], args['label']
            return jsonify({"result":"ok"})
        except Exception as err:
            print(err)
            return jsonify({"error": "unknown error"})
        pass

api.add_resource(PredictionResource, '/predict')
api.add_resource(TrainingResource, '/train')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

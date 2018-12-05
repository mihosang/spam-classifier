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

lstm_model = None
lstm_tok = None
svc_model = None
svc_vect = None

def load_model():
    global lstm_model, lstm_tok, svc_model, svc_vect

    with open('data/model/lstm_model.json', 'r') as handle:
        json_file = handle.read()

    lstm_model = model_from_json(json_file)
    lstm_model.load_weights('data/model/lstm_model.h5')

    svc_model = joblib.load('data/model/C-SVC-model.pkl')
    svc_vect = joblib.load('data/model/vectorizer.joblib')
    with open('data/model/tokenizer.pickle', 'rb') as handle:
        lstm_tok = pickle.load(handle)
    print("load model...")

def lstm_predict(text):

    sequence_matrix = sequence.pad_sequences(lstm_tok.texts_to_sequences(text), maxlen=max_len)
    prediction = lstm_model.predict(sequence_matrix)
    l_predict_proba = []
    for i in range(0, len(text)):
        l_predict_proba.append(prediction[i][0])
    predict = ["phishing" if x > 0.5 else "normal" for x in l_predict_proba]
    l_predict_proba = [str(x) for x in l_predict_proba]

    return predict, l_predict_proba

def svc_predict(text):
    x_test = svc_vect.transform(text)
    predict = svc_model.predict(x_test)
    predict_proba = svc_model.predict_proba(x_test)
    l_predict_proba = []
    for i in range(0, len(text)):
        l_predict_proba.append(predict_proba[i][1])
    predict = ["phishing" if x == 1 else "normal" for x in predict]
    l_predict_proba = [str(x) for x in l_predict_proba]
    return predict, l_predict_proba

class PredictionResource(Resource):

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
                s_predict, s_predict_proba = svc_predict(l_text)
                result = jsonify({"result": s_predict, "probability": s_predict_proba})
            elif model == 'lstm':
                l_predict, l_predict_proba = lstm_predict(l_text)
                result = jsonify({"result": l_predict, "probability": l_predict_proba})
            else:
                s_predict, s_predict_proba = svc_predict(l_text)
                l_predict, l_predict_proba = lstm_predict(l_text)

                result = jsonify(
                    {
                        "svc": {"result": s_predict, "probability": s_predict_proba},
                        "lstm": {"result": l_predict, "probability": l_predict_proba}
                     }
                )
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
    load_model()
    app.run(host='0.0.0.0', port=8000)

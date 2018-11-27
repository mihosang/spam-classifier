from flask import Flask, jsonify
from flask_restful import Resource, Api, request, reqparse
from sklearn.externals import joblib

app = Flask(__name__)
api = Api(app)


clf = joblib.load('./data/C-SVC-model.pkl')
vect = joblib.load('vectorizer.joblib')

class PredictionResource(Resource):


    def post(self):
        try:
            body = request.get_json()
            text = body['text']
            if type(text) is str:
                l_text = [text]
            elif type(text) is list:
                l_text = text
            else:
                return jsonify({"error": "text field is required as string or list of string"})
            x_test = vect.transform(l_text)
            predict = clf.predict(x_test)
            predict_proba = clf.predict_proba(x_test)
            l_predict_proba = []
            for i in range(0,len(l_text)):
                f_predict_proba = "%.2f" % float(predict_proba[i][1]*100) if predict[i] == "phishing" else "%.2f" % float(predict_proba[i][0]*100)
                l_predict_proba.append(f_predict_proba + "%")
            return jsonify({"result": list(predict), "probability": l_predict_proba})
        except:
            return jsonify({"error": "unknown error"})

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

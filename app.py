from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse
from sklearn.externals import joblib

app = Flask(__name__)
api = Api(app)

class PredictionResource(Resource):
    clf = joblib.load('./data/C-SVC-model.pkl')
    vect = joblib.load('vectorizer.joblib')

    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('text')
            args = parser.parse_args()

            text = args['text']

            x_test = self.vect.transform([text])
            predict = self.clf.predict(x_test)
            predict_proba = self.clf.predict_proba(x_test)[0][0]
            f_predict_proba = "%.2f" % float(predict_proba)

            return jsonify({"result": predict[0], "probability": f_predict_proba})
        except:
            return jsonify({"error": "unknown error"})


api.add_resource(PredictionResource, '/predict')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

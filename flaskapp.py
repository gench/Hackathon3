"""
Simple api to serve predictions.
"""
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import json
import numpy as np
import pandas as pd
import data_transformer
from sklearn.externals import joblib

app = Flask(__name__)
api = Api(app)

#{"age":35,"job":"technician","marital":"single","education":"high.school","default":"unknown","housing":"no","loan":"no","contact":"cellular","month":"aug","day_of_week":"fri","duration":1258,"campaign":1,"pdays":999,"previous":0,"poutcome":"nonexistent","emp.var.rate":1.4,"cons.price.idx":93.444,"cons.conf.idx":-36.1,"euribor3m":4.964,"nr.employed":5228.1,"sample_uuid":"74c633e4-65d7-411c-8e0c-575d1153332c"}

#{"sample_uuid":"74c633e4-65d7-411c-8e0c-575d1153332c","probability":0.5,"label":1}

OFFSET = 0.
SIGMA  = 1.

month = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
dow = {"mon":1,"tue":2,"wed":3,"thu":4,"fri":5,"sat":6,"sun":7}

boolean = {'no':0, 'yes':1}

class SimpleModel(Resource):
	def __init__(self):
		self.offset = OFFSET
		self.sigma  = SIGMA
		self.parser = reqparse.RequestParser()
		self.parser.add_argument('sample_uuid', required=True)
		self.parser.add_argument('age', type=int, default=0)
		self.parser.add_argument('month', default='jan')
		self.parser.add_argument('day_of_week', default='mon')
		self.parser.add_argument('duration', type=int, default=0)
		self.parser.add_argument('campaign', type=int, default=0)
		self.parser.add_argument('pdays', type=int, default=0)
		self.parser.add_argument('job', default='unknown')
		self.parser.add_argument('marital', default='unknown') 
                self.parser.add_argument('education', default='unknown')
                self.parser.add_argument('default', default='unknown')
                self.parser.add_argument('housing', default='yes')
                self.parser.add_argument('loan', default='yes')
                self.parser.add_argument('contact', default='cellular')
                self.parser.add_argument('previous', type=int, default=-1)
                self.parser.add_argument('poutcome', default='nonexistent')
                self.parser.add_argument('emp.var.rate', type=float, default=0)
                self.parser.add_argument('cons.price.idx', type=float, default=0)
                self.parser.add_argument('cons.conf.idx', type=float, default=0)
                self.parser.add_argument('euribor3m', type=float, default=0)
                self.parser.add_argument('nr.employed', type=float, default=0)
		self.clf = joblib.load('RF_Classifier.pkl')

	def get(self):
		d = self.parser.parse_args()
		df = pd.DataFrame(d, index=[0])
		dft = data_transformer.data_transformer(df)
		uuid = d.get("sample_uuid", "no uuid")
		prob = self.clf.predict_proba(dft)[0][1]
		labl = 1 if self.clf.predict(dft)[0] else 0
		return json.dumps({"sample_uuid":uuid, "probability":prob, "label":labl})


api.add_resource(SimpleModel, '/api/v1/predict')

if __name__ == '__main__':
     app.run(host="0.0.0.0",port=5000, debug=True)


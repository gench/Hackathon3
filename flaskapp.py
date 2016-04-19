"""
Simple api to serve predictions.
"""
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import json
import numpy as np
import pandas as pd

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
		self.parser.add_argument('age', type=int)
		self.parser.add_argument('month')
		self.parser.add_argument('day_of_week')
		self.parser.add_argument('duration', type=int)
		self.parser.add_argument('campaign', type=int)
		self.parser.add_argument('pdays', type=int)
		self.parser.add_argument('job')
		self.parser.add_argument('marital') 
                self.parser.add_argument('education')
                self.parser.add_argument('default')
                self.parser.add_argument('housing')
                self.parser.add_argument('loan')
                self.parser.add_argument('contact')
                self.parser.add_argument('previous')
                self.parser.add_argument('poutcome')
                self.parser.add_argument('emp.var.rate', type=float)
                self.parser.add_argument('cons.price.idx', type=float)
                self.parser.add_argument('cons.conf.idx', type=float)
                self.parser.add_argument('euribor3m', type=float)
                self.parser.add_argument('nr.employed', type=float)

	def _get(self, js):
		"""Not used. For with json as input"""
		d = json.loads(js)
		uuid = d["sample_uuid"]
		prob = self.logistic(d)
		labl = 1 if prob>0.5 else 0
		return json.dumps({"sample_uuid":uuid, "probability":prob, "label":labl})

	def get(self):
		d = self.parser.parse_args()
		uuid = d.get("sample_uuid", "no uuid")
		prob = self.logistic(d)
		labl = 1 if prob>0.5 else 0
		return json.dumps({"sample_uuid":uuid, "probability":prob, "label":labl})

	def logistic(self, d):
		linear  = 0
		linear += 0 * d.get("age", 0)
		linear += 0 * month.get(d.get("month","none"), 0)
                linear += 0 * dow.get(d.get("day_of_week","none"), 0)
                linear += 0 * boolean.get(d.get("housing", "none"), 0)
                linear += 0 * boolean.get(d.get("loan", "none"), 0)
                #linear += 0 * d.get("emp.var.rate", 0.)
                #linear += 0 * d.get("cons.price.idx", 0.)
                #linear += 0 * d.get("cons.conf.idx", 0.)
                #linear += 0 * d.get("euribor3m", 0.)
                #linear += 0 * d.get("nr.employed", 0.)
                #linear += 0 * d.get("campaign", 0)
                linear += 0 * d.get("", 0)
                linear += 0 * d.get("", 0)
                linear += 0 * d.get("", 0)
                linear += 0 * d.get("", 0)
		return 1./(1 + np.exp(OFFSET-linear/SIGMA))

api.add_resource(SimpleModel, '/predict')

if __name__ == '__main__':
     app.run(host="0.0.0.0",port=5000, debug=True)


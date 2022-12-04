

from collections import OrderedDict

import inspect as inspector

import flask
from flask import Flask
from flask import request
from sklearn.feature_extraction.text import CountVectorizer

import pickle
import tensorflow as tf
from flask_cors import CORS, cross_origin

# if you are using python 3, you should 
import urllib.request

app = Flask(__name__)
CORS(app,resources={r"/getOutput": {"origins": "http://192.168.1.231:5050"}})

model = pickle.load(open('mnb_model.pkl','rb'))
count_vect = pickle.load(open('countvect.pkl','rb'))
@app.route("/getOutput", methods=['POST'])
@cross_origin(origin='*')
def execute_query():
   
    data = request.get_json()
    te = count_vect.transform([data['input']])
    op = model.predict(te)
    
    response = {
       "result":op[0]
    }
    resp = flask.jsonify(response)
    return resp



if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=5050)

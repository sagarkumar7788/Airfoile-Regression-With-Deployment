import pickle
import flask
from flask import Flask,request,app,jsonify,url_for,render_template
from flask import Response
from flask_cors import CORS
import numpy as np
import pandas as pd

app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data'] #using this line we are able to capture the jsion file are avilable that is comming from post
    print(data)
    new_data=[list(data.values())]
    output=model.predict(new_data)[0]
    return jsonify(output)

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()] #using this line we are able to capture the jsion file are avilable that is comming from post using html page
    final_features=[np.array(data)]
    print(data)
    output=model.predict(final_features)[0]
    print(output)
    #output = round(prediction[0],2)
    return render_template('home.html',prediction_text='Airfoil pressure is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
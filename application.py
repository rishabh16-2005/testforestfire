from flask import Flask,request,jsonify,render_template
import pandas as pd 
import pickle 
import numpy as np 
from  sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

model = pickle.load(open('models/model.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))




@app.route("/")
def index():
    return render_template('index.html')
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature = float(request.form.get('Temperature'))
        Rh = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        Ffmc = float(request.form.get('FFMC'))
        Dmc = float(request.form.get('DMC'))
        Isi = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled = standard_scaler.transform([[Temperature,Rh,Ws,Rain,Ffmc,Dmc,Isi,Classes,Region]])
        result = model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])
    else:
        return render_template('home.html')
if __name__=="__main__":
    app.run(host="0.0.0.0")
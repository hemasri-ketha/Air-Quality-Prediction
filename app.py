# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'random_forest_regression_model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        t = int(request.form['t1'])
        tm = int(request.form['t2'])
        tmm = int(request.form['t3'])
        slp = int(request.form['t4'])
        h = int(request.form['t5'])
        vv = float(request.form['t6'])
        v = float(request.form['t7'])
        vm = int(request.form['t8'])
        
        data = np.array([[t, tm, tmm, slp, h, vv, v, vm]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)

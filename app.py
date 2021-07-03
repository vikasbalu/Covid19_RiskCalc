import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/precautions')
def precautions():
    return render_template('precautions.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/symptoms')
def symptoms():
    return render_template('Symptoms.html')

@app.route('/treatments')
def treatments():
    return render_template('treatments.html')
    
@app.route('/eda')
def eda():
    return render_template('eda.html')

@app.route('/predict',methods=['POST'])
def predict():
       
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    if output == 1:
        msg=" High "
    else :
        msg=" Low "
    
    return render_template('result.html', prediction_text='The Risk Of Covid-19 for you is {} '.format(msg))
    
if __name__ == "__main__":
    app.run(debug=True)
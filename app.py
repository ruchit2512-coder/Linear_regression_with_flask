import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load(open('model_weights_simple.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = []
    for x in request.form.values() :
        if x=='Petrol' :
            int_features.append(2)
        elif x=='Diesel':
           int_features.append(3) 
        elif x=='CNG':
           int_features.append(4) 
        elif x=='Dealer':
           int_features.append(2) 
        elif x=='Individual':
           int_features.append(3) 
        elif x=='Manual':
           int_features.append(2) 
        elif x=='Automatic':
           int_features.append(3)
        else :
            int_features.append(int(x)) 
    
    int_features.append(0)
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
# fhfmjhgjh
   #  output = round(prediction[0], 2)
    output = prediction[0]
    return render_template('index.html', prediction_text='price should be $ {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import os 

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

city_dict ={
    'new york city': 1,
    'los angeles': 2,
    'dallas': 3,
    'mountain view': 4, 
    'austin': 5,
    'boston': 6,
    'washington d.c.': 7,
    'san diego': 8
}
gender_dict = {
    'male': 0, 
    'female': 1
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = 'unavailable'
    if prediction == 0:
        output = 'healthy'
    elif prediction == 1:
        output = 'sick'

    return render_template('index.html', prediction_text='Illness status is {}'.format(output))

@app.route('/api/predict')
def ill_predict():
    output1='unavailable'
    city_get = request.args.get('city').lower()
    age = request.args.get('age')
    income = request.args.get('income')
    gender_get = request.args.get('gender').lower()
    if city_get in city_dict.keys():
        city = city_dict.get(city_get)
    else:
        return jsonify({'Illness status': output1}), 400
    if gender_get in gender_dict.keys():
        gender = gender_dict.get(gender_get)
    else:
        return jsonify({'Illness status': output1}), 400
    test_df = pd.DataFrame({'City':[city], 'Age':[age], 'Income':[income], 'Gender':[gender]})
    predicted = model.predict(test_df)
    if predicted == 0:
        output1 = 'healthy'
    elif predicted == 1:
        output1 = 'sick'
    return jsonify({'Illness status': output1})
# if __name__ == "__main__":
#     app.run(debug=True)


port = int(os.environ.get("PORT", 5000))
if __name__ == "__main__":
        app.run(host='0.0.0.0', port=port, debug=True)

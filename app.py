from flask import Flask, render_template, request
import pickle
import numpy as np

model_xgboost = pickle.load(open('xgboost.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST','GET'])
def home():
    if request.method == "POST":
        data1 = request.form['gender']
        data2 = request.form['age']
        data3 = request.form['hypertension']
        data4 = request.form['heartdisease']
        data5 = request.form['evermarried']
        data6 = request.form['worktype']
        data7 = request.form['residencetype']
        data8 = request.form['avgglucoselevel']
        data9 = request.form['bmi']
        data10 = request.form['smokingstatus']

    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10]], dtype=object)
    pred = model_xgboost.predict(arr)
    
    return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug = True)  
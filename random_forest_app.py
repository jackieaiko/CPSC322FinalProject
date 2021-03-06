import os
import pickle
from random import random
from urllib.robotparser import RequestRate

from flask import Flask
from flask import render_template
from flask import request, jsonify, redirect

app = Flask(__name__)
@app.route('/', methods = ['GET', 'POST'])
def index_page():
    prediction = ""
    if request.method == "POST":
        treatment = request.form["treatment"]
        company = request.form["tech_company"]
        wellness_prog = request.form["wellness_program"]
        help = request.form["seek_help"]
        anonymity = request.form["anonymity"]
        leave = request.form["leave"]
        phys_conseq = request.form["phys_health_consequence"]
        coworkers = request.form["coworkers"]
        supervisor = request.form["supervisor"]
        mental_health_int = request.form["mental_health_interview"]
        mental_vs_pyhs = request.form["mental_vs_physical"]
        obs_conseq = request.form["obs_consequence"]
        prediction = predict_random_forest([treatment,company,wellness_prog,help,anonymity,leave,phys_conseq,coworkers,supervisor,mental_health_int,mental_vs_pyhs,obs_conseq])
        print("prediction:", prediction)
        return render_template("success.html",prediction=prediction)
    # goes into templates folder and finds given name
    else:
        return render_template("index.html", prediction=prediction) 

@app.route('/predict', methods=["GET"])
def predict():
    print(request.args)
    treatment = request.args["treatment"]
    company = request.args["tech_company"]
    wellness_prog = request.args["wellness_program"]
    help = request.args["seek_help"]
    anonymity = request.args["anonymity"]
    leave = request.args["leave"]
    phys_conseq = request.args["phys_health_consequence"]
    coworkers = request.args["coworkers"]
    supervisor = request.args["supervisor"]
    mental_health_int = request.args["mental_health_interview"]
    mental_vs_pyhs = request.args["mental_vs_physical"]
    obs_conseq = request.args["obs_consequence"]
    prediction = predict_random_forest([treatment,company,wellness_prog,help,anonymity,leave,phys_conseq,coworkers,supervisor,mental_health_int,mental_vs_pyhs,obs_conseq])
    if prediction is not None:
        # success!
        result = {"prediction": prediction}
        return render_template("success.html",prediction=prediction)
    else:
        return "Error making prediction", 400

def predict_forest(knn_classifier,instance):
    test = ['Yes', 'Yes', 'No', 'Yes', 'Yes', 'Somewhat easy', 'No', 'Some of them', 'Yes', 'No', 'Yes', 'No']
    X_test = [instance]
    prediction = knn_classifier.predict(X_test)
    return prediction[0][0]

def predict_random_forest(unseen_instance):
    # deserialize to object (unpickle)
    infile = open("knn.p", "rb")
    knn_classifier = pickle.load(infile)
    infile.close()
    try:
        return predict_forest(knn_classifier, unseen_instance)
    except:
        return None

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
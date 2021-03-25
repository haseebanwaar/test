# importing the necessary libraries for deployment
from flask import Flask, request, jsonify, render_template, json
import joblib
from pyforest import *


# naming our app as app
from werkzeug.exceptions import HTTPException

app = Flask(__name__)
# app = Flask(__name__, template_folder='templates')
# loading the pickle file for creating the web app
model = joblib.load(open("model.pkl", "rb"))

@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,

    })
    response.content_type = "application/json"
    return response

# defining the different pages of html and specifying the features required to be filled in the html form
@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html")
    # return render_template("index.html")


# creating a function for the prediction model by specifying the parameters and feeding it to the ML model
@app.route("/predict", methods=["POST"])
def predict():
    # specifying our parameters as data type float
    # int_features = [float(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)
    # output = round(prediction[0], 2)
    # return render_template("index.html", prediction_text="flower is {}".format(output))
    return render_template("index.html", prediction_text="flower is flower")


# running the flask app
if __name__ == "__main__":
    app.run(debug=True)

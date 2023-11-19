from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the pre-trained machine learning model
model = joblib.load("heart_disease_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get input from the form
        features = [float(request.form["feature{}".format(i)]) for i in range(13)]

        # Make prediction using the loaded model
        prediction = model.predict([features])

        # Display the result on the result.html page
        return render_template("result.html", prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)

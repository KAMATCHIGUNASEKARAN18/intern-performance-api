from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "Intern Performance Prediction API Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Input features
    features = np.array([[
        data["Completion_Time"],
        data["Feedback_Rating"],
        data["Attendance"],
        data["Productivity"],
        data["Consistency"],
        data["Efficiency"],
        data["Engagement_Level"],
        data["Performance_Index"],
        data["Stability_Score"],
        data["Combined_Score"]
    ]])

    # Scale
    features = scaler.transform(features)

    # Predict
    prediction = model.predict(features)
    result = le.inverse_transform(prediction)

    return jsonify({"Prediction": result[0]})

if __name__ == "__main__":
    app.run(debug=True)
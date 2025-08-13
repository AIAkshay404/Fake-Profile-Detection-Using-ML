import joblib
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load model
model = joblib.load('model.pkl')

# Prediction counts - initializing as zero
prediction_data = {
    "fake": 0,
    "real": 0
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template("about.html")

@app.route("/help")
def help_page():
    return render_template("help.html")

# @app.route("/detection")
# def detection():
#     return render_template("detection.html")

# Get updated chart data
@app.route('/get_chart_data')
def get_chart_data():
    return jsonify(prediction_data)  # Returning dynamic count of fake and real profiles

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    features = [
        int(data["followers"]),
        int(data["following"]),
        int(data["bio_length"]),
        int(data["posts"]),
        int(data["has_profile_pic"]),
        int(data["username_length"]),
        int(data["has_digits"]),
        int(data["account_age"]),
        int(data["is_verified"])
    ]

    prediction = model.predict([features])[0]
    confidence = model.predict_proba([features])[0].max() * 100

    # Update the prediction count dynamically
    if prediction == 1:
        prediction_data["fake"] += 1
    else:
        prediction_data["real"] += 1

    result = "❌Fake Profile" if prediction == 1 else "✅Real Profile"

    return jsonify({
        "result": result,
        "confidence": f"{confidence:.2f}%",
        "fake_count": prediction_data["fake"],
        "real_count": prediction_data["real"]
    })

if __name__ == '__main__':
    app.run(debug=True)

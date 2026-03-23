from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)  # Allow requests from any origin (frontend)

# Load model
model_path = os.path.join(os.path.dirname(__file__), "../ml-model/placement_knn_model.pkl")
model = pickle.load(open(model_path, "rb"))

@app.route('/')
def home():
    return "API is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['features']
        prediction = model.predict([data])
        return jsonify({
            "prediction": int(prediction[0])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Use host="0.0.0.0" for deployment and disable debug mode
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
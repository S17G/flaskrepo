from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)

# --- CORRECTED MODEL LOADING ---
# Get the absolute path of the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Since 'ml-model' is INSIDE 'flask-api', we just join them directly
model_path = os.path.join(BASE_DIR, "ml-model", "placement_knn_model.pkl")

# Safety check to prevent the 'FileNotFoundError' you saw earlier
if not os.path.exists(model_path):
    print(f"❌ CRITICAL ERROR: Model file not found at {model_path}")
    print(f"Current Directory Contents: {os.listdir(BASE_DIR)}")
    model = None
else:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("✅ SUCCESS: Model loaded successfully!")

@app.route('/')
def home():
    return "Placement Predictor API is Live 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500
        
    try:
        data = request.json['features']
        prediction = model.predict([data])
        return jsonify({
            "prediction": int(prediction[0])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
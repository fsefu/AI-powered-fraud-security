from flask import Flask, request, jsonify
import logging
import joblib  # assuming joblib for model persistence

# Initialize the Flask application
app = Flask(__name__)

import joblib

# Correct path based on your structure
MODEL_PATH = "model/MLP_fraud_detection_model.pkl"
model = None

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Setup Logging
logging.basicConfig(level=logging.INFO)
handler = logging.FileHandler("app.log")  # Log to a file named app.log
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

# Define a health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from JSON request
        data = request.get_json()
        logging.info(f"Received data for prediction: {data}")
        
        # Ensure feature order matches model's input expectations
        features = [data[f'feature{i}'] for i in range(1, 31)]
        
        # Make a prediction
        prediction = model.predict([features])[0]
        
        # Convert prediction to standard int type for JSON compatibility
        prediction = int(prediction)
        
        logging.info(f"Prediction result: {prediction}")
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({'error': 'Error during prediction'}), 500
@app.route("/check-model", methods=["GET"])
def check_model():
    if model is None:
        return jsonify({"status": "model not loaded"}), 500
    else:
        return jsonify({"status": "model loaded"}), 200


# Run the application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

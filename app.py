from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

try:
    with open('alert_model.pkl', 'rb') as file:
        model = joblib.load(file)
    le = LabelEncoder()
    le.classes_ = np.array(['CO', 'LPG', 'Smoke', 'Unknown'])
    print("Model and LabelEncoder loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    le = None

@app.route('/')
def home():
    return jsonify({'message': 'Welcome to the Flask server! Use the /predict route for predictions.'})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or le is None:
        return jsonify({'error': 'Model or LabelEncoder not loaded'}), 500

    try:
        data = request.get_json()
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        mq5 = float(data.get('mq5', 0))
        mq7 = float(data.get('mq7', 0))

        features = np.array([[mq5, mq7, temperature, humidity]])
        predictions = model.predict(features)[0]
        alert_pred = predictions[0]
        gas_pred_encoded = predictions[1]
        gas_pred = le.inverse_transform([int(gas_pred_encoded)])[0]

        alerts = [
            f"Alert: {alert_pred}, Suspected Gas: {gas_pred}, Sensor Values - MQ5: {mq5} ppm, MQ7: {mq7} ppm, Temp: {temperature}°C, Humidity: {humidity}%"
        ]
        danger = bool(alert_pred == 1)  # Convert to Python bool

        return jsonify({
            'mq5_pred': mq5,
            'mq7_pred': mq7,
            'alert_pred': int(alert_pred),
            'prediction': int(alert_pred),
            'suspected_gas': gas_pred,
            'alerts': alerts,
            'danger': danger
        })
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '_main_':
    app.run(host='0.0.0.0', port=5000)
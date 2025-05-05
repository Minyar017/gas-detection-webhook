from flask import Flask, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, db, firestore
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import threading
import time

app = Flask(__name__)
CORS(app)

# ----------------- Load ML Model -----------------
try:
    with open('alert_model.pkl', 'rb') as file:
        model = joblib.load(file)
    le = LabelEncoder()
    le.classes_ = np.array(['CO', 'LPG', 'Smoke', 'Unknown'])
    print("✅ Model and LabelEncoder loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    le = None

# ----------------- Initialize Firebase -----------------
try:
    cred = credentials.Certificate('serviceAccountKey.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://projet-fin-d-etude-4632f-default-rtdb.firebaseio.com/'  # Replace with your project ID
    })
    firestore_db = firestore.client()
    print("✅ Firebase initialized")
except Exception as e:
    print(f"❌ Error initializing Firebase: {e}")

# ----------------- Background Prediction Function -----------------
def monitor_sensor_data():
    last_processed_key = None
    while True:
        try:
            ref = db.reference('sensor_data')
            all_data = ref.get()

            if not all_data:
                print("ℹ️ No sensor data found.")
                time.sleep(1)
                continue

            # Get last inserted item
            last_key = list(all_data.keys())[-1]
            if last_key == last_processed_key:
                time.sleep(5)
                continue  # Skip if same data

            last_processed_key = last_key
            latest = all_data[last_key]
            print(f"📡 New data received: {latest}")

            # Extract fields safely
            temperature = float(latest.get('temperature', 0))
            humidity = float(latest.get('humidity', 0))
            mq5 = float(latest.get('mq5', 0))
            mq7 = float(latest.get('mq7', 0))

            # Reorder features to match model input
            features = np.array([[humidity, mq5, mq7, temperature]])
            predictions = model.predict(features)[0]
            alert_pred = int(predictions[0])

            if alert_pred == 1:
                gas_encoded = int(predictions[1])
                gas_name = le.inverse_transform([gas_encoded])[0]

                alert_data = {
                    'alert': alert_pred,
                    'suspected_gas': gas_name,
                    'timestamp': firestore.SERVER_TIMESTAMP,
                    'sensor_values': {
                        'mq5': mq5,
                        'mq7': mq7,
                        'temperature': temperature,
                        'humidity': humidity
                    }
                }

                # Save to Firestore only if alert == 1
                firestore_db.collection('alerts').add(alert_data)
                print(f"🚨 Alert detected and saved to Firestore: {alert_data}")
            else:
                print("✅ No alert detected. Data not saved.")

        except Exception as e:
            print(f"❌ Error during monitoring: {e}")
        time.sleep(5)  # Wait before checking again

# ----------------- Flask Routes -----------------
@app.route('/')
def home():
    return jsonify({'message': 'Flask server is running. Predictions are processed automatically.'})

# ----------------- Start Background Thread -----------------
if model and le:
    threading.Thread(target=monitor_sensor_data, daemon=True).start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

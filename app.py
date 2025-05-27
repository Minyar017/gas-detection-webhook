from flask import Flask, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, db, firestore
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import threading
import time
import os
import json

app = Flask(__name__)
CORS(app)

# ----------------- Load ML Model -----------------
try:
    with open('alert_model.pkl', 'rb') as file:
        all_models = joblib.load(file)
    
    # Modèle multi-sortie (pour prédictions simultanées)
    multi_output_model = all_models['multi_output_model']
    
    # Modèles spécialisés pour 'alert'
    alert_models = all_models['classifiers_alert']
    
    # Modèle Naive Bayes pour 'suspected_gas' uniquement
    gas_model = all_models['classifiers_suspected_gas']['Naive Bayes']
    
    # Outils de préprocessing
    scaler = all_models['scaler']
    le = all_models['label_encoder']
    
    print("✅ Tous les modèles chargés avec succès:")
    print(f"   - Modèle multi-sortie: {type(multi_output_model).__name__}")
    print(f"   - Modèles pour alert: {list(alert_models.keys())}")
    print(f"   - Modèle pour suspected_gas: Naive Bayes")
    print(f"   - Scaler et LabelEncoder chargés")
    
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle: {e}")
    multi_output_model = alert_models = gas_model = scaler = le = None

# ----------------- Initialize Firebase from ENV -----------------
try:
    firebase_credentials = os.getenv("FIREBASE_CREDENTIALS")
    if not firebase_credentials:
        raise ValueError("FIREBASE_CREDENTIALS environment variable not set")

    cred_dict = json.loads(firebase_credentials)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://projet-fin-d-etude-4632f-default-rtdb.firebaseio.com/'
    })
    firestore_db = firestore.client()
    db_ref = db.reference('sensor_data')
    print("✅ Firebase initialized from environment variable")
except Exception as e:
    print(f"❌ Error initializing Firebase: {str(e)}")
    firestore_db = None
    db_ref = None

# ----------------- Prediction Functions -----------------
def predict_with_multi_output(features):
    """
    Utilise le modèle multi-sortie pour prédire alert et suspected_gas simultanément
    """
    try:
        features_scaled = scaler.transform(features)
        predictions = multi_output_model.predict(features_scaled)[0]
        
        alert_pred = int(predictions[0])
        gas_encoded = int(predictions[1])
        gas_name = le.inverse_transform([gas_encoded])[0]
        
        return alert_pred, gas_name
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction multi-sortie: {e}")
        return None, None

def predict_with_specialized_models(features):
    """
    Utilise les modèles spécialisés: Logistic Regression pour alert + Naive Bayes pour gas
    """
    try:
        features_scaled = scaler.transform(features)
        
        # Prédiction d'alerte avec Logistic Regression (ou autre modèle de votre choix)
        alert_pred = int(alert_models['Logistic Regression'].predict(features_scaled)[0])
        
        # Prédiction de gaz avec Naive Bayes uniquement
        gas_encoded = int(gas_model.predict(features_scaled)[0])
        gas_name = le.inverse_transform([gas_encoded])[0]
        
        return alert_pred, gas_name
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction spécialisée: {e}")
        return None, None

# ----------------- Background Prediction Function -----------------
def monitor_sensor_data():
    last_processed_key = None
    while True:
        try:
            if db_ref is None or firestore_db is None:
                print("⚠ Firebase not initialized. Skipping monitoring.")
                time.sleep(5)
                continue

            if multi_output_model is None or gas_model is None or scaler is None or le is None:
                print("⚠ Models not loaded. Skipping monitoring.")
                time.sleep(5)
                continue

            all_data = db_ref.get()
            if not all_data:
                print("ℹ No sensor data found.")
                time.sleep(1)
                continue

            last_key = list(all_data.keys())[-1]
            if last_key == last_processed_key:
                time.sleep(1)
                continue

            last_processed_key = last_key
            latest = all_data[last_key]
            print(f"📡 New data received: {latest}")

            temperature = float(latest.get('temperature', 0))
            humidity = float(latest.get('humidity', 0))
            mq5 = float(latest.get('mq5', 0))
            mq7 = float(latest.get('mq7', 0))

            features = np.array([[humidity, mq5, mq7, temperature]])
            
            # OPTION 1: Utiliser le modèle multi-sortie
            alert_pred, gas_name = predict_with_multi_output(features)
            
            # OPTION 2: Utiliser les modèles spécialisés (commenté)
            # alert_pred, gas_name = predict_with_specialized_models(features)
            
            if alert_pred is None or gas_name is None:
                print("❌ Erreur lors de la prédiction. Données ignorées.")
                continue

            print(f"🔮 Prédiction: Alert={alert_pred}, Gas={gas_name}")

            if alert_pred == 1:
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

                firestore_db.collection('alerts').add(alert_data)
                print(f"🚨 Alert detected and saved to Firestore: {alert_data}")
            else:
                print("✅ No alert detected. Data not saved.")

        except Exception as e:
            print(f"❌ Error during monitoring: {e}")
        time.sleep(5)

# ----------------- Flask Routes -----------------
@app.route('/')
def home():
    return jsonify({
        'message': 'Flask server is running. Predictions are processed automatically.',
        'models_loaded': {
            'multi_output_model': multi_output_model is not None,
            'alert_models': alert_models is not None,
            'gas_model': gas_model is not None,
            'scaler': scaler is not None,
            'label_encoder': le is not None
        }
    })

@app.route('/predict', methods=['POST'])
def manual_predict():
    """
    Route pour tester manuellement les prédictions
    """
    try:
        from flask import request
        
        if multi_output_model is None or gas_model is None or scaler is None or le is None:
            return jsonify({'error': 'Models not loaded'}), 500
        
        data = request.get_json()
        humidity = float(data.get('humidity', 0))
        mq5 = float(data.get('mq5', 0))
        mq7 = float(data.get('mq7', 0))
        temperature = float(data.get('temperature', 0))
        
        features = np.array([[humidity, mq5, mq7, temperature]])
        
        # Prédictions avec les deux approches
        alert_multi, gas_multi = predict_with_multi_output(features)
        alert_spec, gas_spec = predict_with_specialized_models(features)
        
        return jsonify({
            'multi_output_prediction': {
                'alert': alert_multi,
                'suspected_gas': gas_multi
            },
            'specialized_prediction': {
                'alert': alert_spec,
                'suspected_gas': gas_spec
            },
            'input_data': {
                'humidity': humidity,
                'mq5': mq5,
                'mq7': mq7,
                'temperature': temperature
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ----------------- Start Background Thread -----------------
if multi_output_model and gas_model and scaler and le:
    print("🚀 Starting background monitoring thread...")
    threading.Thread(target=monitor_sensor_data, daemon=True).start()
else:
    print("❌ Cannot start monitoring: Models not properly loaded")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

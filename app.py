from flask import Flask, jsonify, request
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
import traceback

app = Flask(__name__)
CORS(app)

# Global variables for models and Firebase
multi_output_model = None
alert_models = None
gas_model = None
scaler = None
le = None
firestore_db = None
db_ref = None
monitoring_active = False

# ----------------- Load ML Model -----------------
def load_models():
    """Load all ML models with comprehensive error handling"""
    global multi_output_model, alert_models, gas_model, scaler, le
    
    try:
        print("🔄 Loading ML models...")
        
        # Check if model file exists
        if not os.path.exists('alert_model.pkl'):
            raise FileNotFoundError("alert_model.pkl not found in current directory")
        
        with open('alert_model.pkl', 'rb') as file:
            all_models = joblib.load(file)
        
        # Load multi-output model
        if 'multi_output_model' in all_models:
            multi_output_model = all_models['multi_output_model']
            print(f"   ✅ Multi-output model loaded: {type(multi_output_model).__name__}")
        else:
            print("   ⚠️ Multi-output model not found in saved models")
        
        # Load specialized models for alert
        if 'classifiers_alert' in all_models:
            alert_models = all_models['classifiers_alert']
            print(f"   ✅ Alert models loaded: {list(alert_models.keys())}")
            
            # Verify each model is accessible
            for name, model in alert_models.items():
                try:
                    # Test if model has predict method
                    if hasattr(model, 'predict'):
                        print(f"      ✅ {name}: Ready")
                    else:
                        print(f"      ❌ {name}: No predict method")
                except Exception as e:
                    print(f"      ❌ {name}: Error - {e}")
        else:
            print("   ❌ Alert models not found in saved models")
        
        # Load gas detection model
        if 'classifiers_suspected_gas' in all_models and 'Naive Bayes' in all_models['classifiers_suspected_gas']:
            gas_model = all_models['classifiers_suspected_gas']['Naive Bayes']
            print(f"   ✅ Gas model loaded: Naive Bayes")
        else:
            print("   ❌ Gas model (Naive Bayes) not found")
        
        # Load preprocessing tools
        if 'scaler' in all_models:
            scaler = all_models['scaler']
            print(f"   ✅ Scaler loaded: {type(scaler).__name__}")
        else:
            print("   ❌ Scaler not found")
        
        if 'label_encoder' in all_models:
            le = all_models['label_encoder']
            print(f"   ✅ Label encoder loaded with classes: {list(le.classes_)}")
        else:
            print("   ❌ Label encoder not found")
        
        # Verify all critical components are loaded
        if all([multi_output_model, alert_models, gas_model, scaler, le]):
            print("🎉 All models loaded successfully!")
            return True
        else:
            missing = []
            if not multi_output_model: missing.append("multi_output_model")
            if not alert_models: missing.append("alert_models")
            if not gas_model: missing.append("gas_model")
            if not scaler: missing.append("scaler")
            if not le: missing.append("label_encoder")
            print(f"❌ Missing components: {missing}")
            return False
            
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print(f"❌ Traceback: {traceback.format_exc()}")
        return False

# ----------------- Initialize Firebase -----------------
def initialize_firebase():
    """Initialize Firebase with comprehensive error handling"""
    global firestore_db, db_ref
    
    try:
        print("🔄 Initializing Firebase...")
        
        firebase_credentials = os.getenv("FIREBASE_CREDENTIALS")
        if not firebase_credentials:
            raise ValueError("FIREBASE_CREDENTIALS environment variable not set")

        cred_dict = json.loads(firebase_credentials)
        cred = credentials.Certificate(cred_dict)
        
        # Initialize Firebase Admin SDK
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://projet-fin-d-etude-4632f-default-rtdb.firebaseio.com/'
        })
        
        # Initialize Firestore and Realtime Database
        firestore_db = firestore.client()
        db_ref = db.reference('sensor_data')
        
        print("✅ Firebase initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing Firebase: {str(e)}")
        print(f"❌ Traceback: {traceback.format_exc()}")
        return False

# ----------------- Prediction Functions -----------------
def predict_with_multi_output(features):
    """Use multi-output model for simultaneous predictions"""
    try:
        if not all([multi_output_model, scaler, le]):
            raise ValueError("Multi-output model components not available")
        
        features_scaled = scaler.transform(features)
        predictions = multi_output_model.predict(features_scaled)[0]
        
        alert_pred = int(predictions[0])
        gas_encoded = int(predictions[1])
        gas_name = le.inverse_transform([gas_encoded])[0]
        
        return alert_pred, gas_name
        
    except Exception as e:
        print(f"❌ Error in multi-output prediction: {e}")
        return None, None

def predict_with_specialized_models(features, preferred_alert_model='Logistic Regression'):
    """Use specialized models with fallback options"""
    try:
        if not all([alert_models, gas_model, scaler, le]):
            raise ValueError("Specialized model components not available")
        
        features_scaled = scaler.transform(features)
        
        # Try preferred model first, then fallback to available models
        alert_pred = None
        if preferred_alert_model in alert_models:
            try:
                alert_pred = int(alert_models[preferred_alert_model].predict(features_scaled)[0])
                print(f"✅ Alert prediction using {preferred_alert_model}: {alert_pred}")
            except Exception as e:
                print(f"⚠️ Error with {preferred_alert_model}: {e}")
        
        # Fallback to first available model
        if alert_pred is None:
            for model_name, model in alert_models.items():
                try:
                    alert_pred = int(model.predict(features_scaled)[0])
                    print(f"✅ Alert prediction using fallback {model_name}: {alert_pred}")
                    break
                except Exception as e:
                    print(f"⚠️ Error with fallback {model_name}: {e}")
                    continue
        
        if alert_pred is None:
            raise ValueError("No alert model could make prediction")
        
        # Gas prediction with Naive Bayes
        gas_encoded = int(gas_model.predict(features_scaled)[0])
        gas_name = le.inverse_transform([gas_encoded])[0]
        
        return alert_pred, gas_name
        
    except Exception as e:
        print(f"❌ Error in specialized prediction: {e}")
        return None, None

# ----------------- Background Monitoring -----------------
def monitor_sensor_data():
    """Background thread to monitor sensor data and make predictions"""
    global monitoring_active
    last_processed_key = None
    monitoring_active = True
    
    print("🚀 Background monitoring started")
    
    while monitoring_active:
        try:
            # Check if all components are available
            if not all([db_ref, firestore_db]):
                print("⚠️ Firebase not initialized. Retrying in 10 seconds...")
                time.sleep(10)
                continue

            if not all([multi_output_model, scaler, le]):
                print("⚠️ Models not loaded. Retrying in 10 seconds...")
                time.sleep(10)
                continue

            # Get latest sensor data
            all_data = db_ref.get()
            if not all_data:
                print("ℹ️ No sensor data found. Waiting...")
                time.sleep(5)
                continue

            # Process only new data
            last_key = list(all_data.keys())[-1]
            if last_key == last_processed_key:
                time.sleep(2)
                continue

            last_processed_key = last_key
            latest = all_data[last_key]
            print(f"📡 Processing new data: {latest}")

            # Extract sensor values with validation
            try:
                temperature = float(latest.get('temperature', 0))
                humidity = float(latest.get('humidity', 0))
                mq5 = float(latest.get('mq5', 0))
                mq7 = float(latest.get('mq7', 0))
            except (ValueError, TypeError) as e:
                print(f"❌ Invalid sensor data format: {e}")
                continue

            # Prepare features for prediction
            features = np.array([[humidity, mq5, mq7, temperature]])
            
            # Make predictions (try multi-output first, then specialized)
            alert_pred, gas_name = predict_with_multi_output(features)
            
            if alert_pred is None:
                print("⚠️ Multi-output prediction failed, trying specialized models...")
                alert_pred, gas_name = predict_with_specialized_models(features)
            
            if alert_pred is None or gas_name is None:
                print("❌ All prediction methods failed. Skipping this data point.")
                continue

            print(f"🔮 Prediction successful: Alert={alert_pred}, Gas={gas_name}")

            # Save alert if detected
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
                    },
                    'prediction_method': 'multi_output' if predict_with_multi_output(features)[0] is not None else 'specialized'
                }

                try:
                    firestore_db.collection('alerts').add(alert_data)
                    print(f"🚨 ALERT SAVED: {gas_name} detected!")
                except Exception as e:
                    print(f"❌ Error saving alert to Firestore: {e}")
            else:
                print("✅ No alert detected")

        except Exception as e:
            print(f"❌ Error in monitoring loop: {e}")
            print(f"❌ Traceback: {traceback.format_exc()}")
        
        time.sleep(3)  # Check every 3 seconds

# ----------------- Flask Routes -----------------
@app.route('/')
def home():
    """Health check endpoint"""
    models_status = {
        'multi_output_model': multi_output_model is not None,
        'alert_models': alert_models is not None and len(alert_models) > 0,
        'gas_model': gas_model is not None,
        'scaler': scaler is not None,
        'label_encoder': le is not None
    }
    
    firebase_status = {
        'firestore': firestore_db is not None,
        'realtime_db': db_ref is not None
    }
    
    return jsonify({
        'message': 'Gas Detection API is running',
        'status': 'healthy' if all(models_status.values()) and all(firebase_status.values()) else 'degraded',
        'models_loaded': models_status,
        'firebase_initialized': firebase_status,
        'monitoring_active': monitoring_active,
        'available_alert_models': list(alert_models.keys()) if alert_models else [],
        'gas_classes': list(le.classes_) if le else []
    })

@app.route('/predict', methods=['POST'])
def manual_predict():
    """Manual prediction endpoint for testing"""
    try:
        # Validate models are loaded
        if not all([scaler, le]):
            return jsonify({'error': 'Core models (scaler/label_encoder) not loaded'}), 500
        
        # Get and validate input data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        try:
            humidity = float(data.get('humidity', 0))
            mq5 = float(data.get('mq5', 0))
            mq7 = float(data.get('mq7', 0))
            temperature = float(data.get('temperature', 0))
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid sensor data format'}), 400
        
        features = np.array([[humidity, mq5, mq7, temperature]])
        
        results = {
            'input_data': {
                'humidity': humidity,
                'mq5': mq5,
                'mq7': mq7,
                'temperature': temperature
            },
            'predictions': {}
        }
        
        # Try multi-output prediction
        if multi_output_model:
            alert_multi, gas_multi = predict_with_multi_output(features)
            if alert_multi is not None:
                results['predictions']['multi_output'] = {
                    'alert': alert_multi,
                    'suspected_gas': gas_multi,
                    'status': 'success'
                }
            else:
                results['predictions']['multi_output'] = {'status': 'failed'}
        
        # Try specialized models prediction
        if alert_models and gas_model:
            alert_spec, gas_spec = predict_with_specialized_models(features)
            if alert_spec is not None:
                results['predictions']['specialized'] = {
                    'alert': alert_spec,
                    'suspected_gas': gas_spec,
                    'status': 'success'
                }
            else:
                results['predictions']['specialized'] = {'status': 'failed'}
        
        # Check if any prediction succeeded
        if not any(pred.get('status') == 'success' for pred in results['predictions'].values()):
            return jsonify({'error': 'All prediction methods failed', 'results': results}), 500
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/status')
def detailed_status():
    """Detailed system status endpoint"""
    return jsonify({
        'models': {
            'multi_output_available': multi_output_model is not None,
            'alert_models_count': len(alert_models) if alert_models else 0,
            'alert_models_list': list(alert_models.keys()) if alert_models else [],
            'gas_model_available': gas_model is not None,
            'scaler_available': scaler is not None,
            'label_encoder_available': le is not None,
            'gas_classes': list(le.classes_) if le else []
        },
        'firebase': {
            'firestore_connected': firestore_db is not None,
            'realtime_db_connected': db_ref is not None
        },
        'monitoring': {
            'active': monitoring_active,
            'thread_alive': monitoring_active
        }
    })

# ----------------- Application Initialization -----------------
def initialize_app():
    """Initialize the entire application"""
    print("🚀 Initializing Gas Detection API...")
    
    # Load ML models
    models_loaded = load_models()
    
    # Initialize Firebase
    firebase_initialized = initialize_firebase()
    
    # Start background monitoring if everything is ready
    if models_loaded and firebase_initialized:
        print("🎯 Starting background monitoring thread...")
        monitoring_thread = threading.Thread(target=monitor_sensor_data, daemon=True)
        monitoring_thread.start()
        print("✅ Application fully initialized!")
    else:
        print("⚠️ Application initialized with limited functionality")
        if not models_loaded:
            print("   - Models not loaded properly")
        if not firebase_initialized:
            print("   - Firebase not initialized")

if __name__ == '__main__':
    # Initialize the application
    initialize_app()
    
    # Start Flask server
    print("🌐 Starting Flask server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

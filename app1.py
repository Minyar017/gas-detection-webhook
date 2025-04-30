from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from firebase_admin import credentials, db, initialize_app

app = Flask(__name__)
CORS(app)  # Enable CORS for Dialogflow requests

# Load Firebase credentials from environment variable
try:
    firebase_credentials = os.getenv("FIREBASE_CREDENTIALS")
    if not firebase_credentials:
        raise ValueError("FIREBASE_CREDENTIALS environment variable not set")

    cred_dict = json.loads(firebase_credentials)
    cred = credentials.Certificate(cred_dict)
    initialize_app(cred, {
        'databaseURL': 'https://projet-fin-d-etude-4632f-default-rtdb.firebaseio.com/'
    })
    print("Firebase initialized successfully")
except Exception as e:
    print(f"Failed to initialize Firebase: {str(e)}")
    raise ValueError(f"Failed to initialize Firebase: {str(e)}")

@app.route('/')
def home():
    return "Flask app is running!"

@app.route('/process_command', methods=['POST'])
def process_command():
    if not request.is_json:
        return jsonify({"fulfillmentText": "Erreur: Content-Type must be application/json"}), 415

    try:
        data = request.get_json()
        intent = data['queryResult']['intent']['displayName']
        if intent == 'get_co_level':
            # Replace with your logic to fetch CO level from Firebase
            co_level = 50  # Example value; fetch from Firebase in reality
            return jsonify({"fulfillmentText": f"Le niveau de CO actuel est de {co_level} ppm."})
        return jsonify({"fulfillmentText": "Intent not recognized."})
    except Exception as e:
        return jsonify({"fulfillmentText": f"Erreur: Impossible de parser le JSON: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 10000)))
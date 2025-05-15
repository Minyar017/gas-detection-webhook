from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, auth, firestore
import os
import json

app = Flask(__name__)
CORS(app)  # Important pour accepter les requêtes depuis Flutter web

# Initialisation Firebase
firebase_json = os.getenv('FIREBASE_SERVICE_ACCOUNT_JSON')
cred = credentials.Certificate(json.loads(firebase_json))
firebase_admin.initialize_app(cred)
db = firestore.client()

@app.route("/delete_user_endpoint", methods=["POST"])
def delete_user():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "Non autorisé"}), 401

    id_token = auth_header.split("Bearer ")[1]
    try:
        decoded_token = auth.verify_id_token(id_token)
        admin_uid = decoded_token['uid']

        # Vérifie si c’est un admin
        admin_doc = db.collection('admins').document(admin_uid).get()
        if not admin_doc.exists or admin_doc.to_dict().get('role') != 'admin':
            return jsonify({"error": "Accès réservé aux admins"}), 403

        user_id = request.json.get('user_id')
        if not user_id:
            return jsonify({"error": "user_id manquant"}), 400

        # Supprimer de Firebase Auth
        auth.delete_user(user_id)

        # Supprimer de Firestore
        for collection in ['users', 'userLogs', 'admins']:
            ref = db.collection(collection).document(user_id)
            if ref.get().exists:
                ref.delete()

        return jsonify({"message": f"Utilisateur {user_id} supprimé"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

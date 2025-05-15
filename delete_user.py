from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import auth, firestore, credentials
import os
import json

app = Flask(__name__)
CORS(app)  # Autorise CORS pour toutes les origines (à restreindre en production)

# Charger les identifiants Firebase depuis variable d'environnement
service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
if not service_account_json:
    raise Exception("La variable d'environnement FIREBASE_SERVICE_ACCOUNT_JSON n'est pas définie")

cred_dict = json.loads(service_account_json)
cred = credentials.Certificate(cred_dict)

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

@app.route("/delete_user_endpoint", methods=["POST"])
def delete_user():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({"error": "Token manquant ou invalide"}), 401

    id_token = auth_header.split('Bearer ')[1]

    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        print(f"Utilisateur authentifié uid={uid}")

        admin_doc = db.collection('admins').document(uid).get()
        if not admin_doc.exists or admin_doc.to_dict().get('role') != 'admin':
            return jsonify({"error": "Accès réservé aux admins"}), 403

    except Exception as e:
        print(f"Erreur d'authentification: {str(e)}")
        return jsonify({"error": "Erreur d'authentification"}), 401

    data = request.get_json()
    if not data or "user_id" not in data:
        return jsonify({"error": "user_id requis"}), 400

    user_id = data["user_id"]

    try:
        auth.get_user(user_id)
        print(f"Suppression utilisateur {user_id}")

        auth.delete_user(user_id)
        print(f"Utilisateur {user_id} supprimé de Firebase Auth")

        for collection in ['users', 'userLogs', 'admins']:
            doc_ref = db.collection(collection).document(user_id)
            if doc_ref.get().exists:
                doc_ref.delete()
                print(f"Document supprimé de {collection} pour {user_id}")

        return jsonify({"message": f"Utilisateur {user_id} supprimé avec succès"}), 200

    except auth.UserNotFoundError:
        return jsonify({"error": f"Utilisateur {user_id} non trouvé"}), 404
    except Exception as e:
        print(f"Erreur suppression: {str(e)}")
        return jsonify({"error": "Erreur serveur interne"}), 500

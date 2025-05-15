from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, auth, firestore
import json
import os

app = Flask(__name__)

# Initialiser Firebase Admin à partir d'une variable d'environnement
firebase_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
cred = credentials.Certificate(json.loads(firebase_json))
firebase_admin.initialize_app(cred)
db = firestore.client()

@app.route('/delete_user_endpoint', methods=['POST'])
def delete_user_endpoint():
    try:
        # Authentification via le token envoyé depuis Flutter
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Token manquant"}), 401

        id_token = auth_header.split("Bearer ")[1]
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token["uid"]

        # Vérifier si admin
        admin_doc = db.collection("admins").document(uid).get()
        if not admin_doc.exists or admin_doc.to_dict().get("role") != "admin":
            return jsonify({"error": "Accès refusé"}), 403

        # Récupérer l'ID de l'utilisateur à supprimer
        body = request.get_json()
        user_id = body.get("user_id")
        if not user_id:
            return jsonify({"error": "user_id requis"}), 400

        # Supprimer dans Firebase Auth
        auth.delete_user(user_id)

        # Supprimer dans Firestore
        for col in ['users', 'userLogs', 'admins']:
            doc_ref = db.collection(col).document(user_id)
            if doc_ref.get().exists:
                doc_ref.delete()

        return jsonify({"message": f"Utilisateur {user_id} supprimé avec succès"}), 200

    except auth.UserNotFoundError:
        return jsonify({"error": "Utilisateur non trouvé"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

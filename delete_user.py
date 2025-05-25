from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, auth, firestore
import os
import json
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Autoriser les requêtes cross-origin (utile pour Flutter web)

# Initialisation Firebase
try:
    firebase_json = os.getenv('FIREBASE_SERVICE_ACCOUNT_JSON')
    if not firebase_json:
        logger.error("Variable d'environnement FIREBASE_SERVICE_ACCOUNT_JSON non définie")
        raise ValueError("Configuration Firebase manquante")
    
    # Remplacer les \n échappés par des vrais sauts de ligne
    parsed_json = json.loads(firebase_json.replace("\\n", "\n"))
    cred = credentials.Certificate(parsed_json)
    
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    logger.info("Firebase initialisé avec succès")
except Exception as e:
    logger.error(f"Erreur d'initialisation Firebase: {e}")
    # Si Firebase échoue à s'initialiser, on gère ça plus bas dans les endpoints

@app.route("/delete_user_endpoint", methods=["POST"])
def delete_user():
    logger.info("Requête de suppression d'utilisateur reçue")
    
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.warning("Tentative d'accès sans token d'authentification")
        return jsonify({"error": "Non autorisé"}), 401

    request_data = request.get_json()
    if not request_data:
        logger.warning("Données JSON manquantes dans la requête")
        return jsonify({"error": "Données JSON requises"}), 400

    user_id = request_data.get('user_id')
    if not user_id:
        logger.warning("user_id manquant dans la requête")
        return jsonify({"error": "user_id manquant"}), 400

    logger.info(f"Tentative de suppression pour user_id: {user_id}")

    id_token = auth_header.split("Bearer ")[1]
    try:
        # Vérifie l'authenticité du token
        decoded_token = auth.verify_id_token(id_token)
        admin_uid = decoded_token['uid']
        logger.info(f"Token vérifié pour admin_uid: {admin_uid}")

        # Vérifie que l'utilisateur est admin
        admin_doc = db.collection('admins').document(admin_uid).get()
        if not admin_doc.exists or admin_doc.to_dict().get('role') != 'admin':
            logger.warning(f"Tentative d'accès par un non-admin: {admin_uid}")
            return jsonify({"error": "Accès réservé aux admins"}), 403

        # Vérifie si l'utilisateur existe dans Firebase Auth
        try:
            user = auth.get_user(user_id)
            logger.info(f"Utilisateur trouvé dans Auth: {user.uid}")
        except auth.UserNotFoundError:
            logger.warning(f"Utilisateur non trouvé dans Auth: {user_id}")
            return jsonify({"error": f"Utilisateur {user_id} non trouvé dans Auth"}), 404

        # Supprime l'utilisateur de Firebase Auth
        try:
            auth.delete_user(user_id)
            logger.info(f"Utilisateur supprimé de Auth: {user_id}")
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de Auth: {e}")
            return jsonify({"error": f"Erreur de suppression Auth: {str(e)}"}), 500

        # Supprime les documents Firestore liés
        collections_deleted = []
        for collection in ['users', 'userLogs', 'admins']:
            ref = db.collection(collection).document(user_id)
            if ref.get().exists:
                ref.delete()
                collections_deleted.append(collection)
                logger.info(f"Document supprimé de {collection}: {user_id}")

        response_message = {
            "message": f"Utilisateur {user_id} supprimé avec succès",
            "details": {
                "auth_deleted": True,
                "collections_deleted": collections_deleted
            }
        }

        logger.info(f"Suppression réussie - Détails: {response_message}")
        return jsonify(response_message), 200

    except auth.InvalidIdTokenError:
        logger.error(f"Token invalide: {id_token[:20]}...")
        return jsonify({"error": "Token d'authentification invalide"}), 401
    except Exception as e:
        logger.error(f"Erreur inattendue: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

import os
import firebase_admin
from firebase_admin import auth, credentials, firestore
from flask import Flask, request, jsonify
from functools import wraps

# Initialisation sécurisée de Firebase avec les variables d'environnement
firebase_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')

if not firebase_json:
    raise ValueError("La variable d'environnement FIREBASE_SERVICE_ACCOUNT_JSON est manquante.")

cred = credentials.Certificate(eval(firebase_json))
firebase_admin.initialize_app(cred)

db = firestore.client()

# Création de l'application Flask
app = Flask(__name__)

# Vérification du token envoyé par le client Flutter
def verify_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'Authorization header is missing'}), 401

        try:
            id_token = auth_header.split('Bearer ')[1]
            decoded_token = auth.verify_id_token(id_token)
            request.user = decoded_token
        except Exception as e:
            return jsonify({'error': f'Token invalide: {str(e)}'}), 401

        return f(*args, **kwargs)

    return decorated_function

# Endpoint de suppression d'un utilisateur
@app.route('/delete_user_endpoint', methods=['POST'])
@verify_token
def delete_user():
    try:
        data = request.get_json()
        user_id = data.get('user_id')

        if not user_id:
            return jsonify({'error': 'user_id est requis'}), 400

        # Supprimer l'utilisateur de Firebase Authentication
        auth.delete_user(user_id)

        # Supprimer l'utilisateur de Firestore
        db.collection('users').document(user_id).delete()

        return jsonify({'message': 'Utilisateur supprimé avec succès'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

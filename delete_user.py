from flask import Flask, request, jsonify
from flask_cors import CORS
from firebase_admin import initialize_app, auth, firestore

# Initialisation Firebase Admin
initialize_app()

app = Flask(__name__)
CORS(app)  # Autorise toutes les origines (à restreindre en production)

@app.route('/delete_user_endpoint', methods=['POST', 'OPTIONS'])
def delete_user_endpoint():
    if request.method == 'OPTIONS':
        # Réponse pour CORS préflight
        return '', 200

    # Vérifier l’en-tête Authorization
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Non autorisé, token manquant'}), 401

    id_token = auth_header.split('Bearer ')[1]
    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']

        # Vérifier rôle admin dans Firestore
        db = firestore.client()
        admin_doc = db.collection('admins').document(uid).get()
        if not admin_doc.exists or admin_doc.to_dict().get('role') != 'admin':
            return jsonify({'error': 'Accès réservé aux admins'}), 403

    except Exception as e:
        return jsonify({'error': f'Erreur d\'authentification: {str(e)}'}), 401

    data = request.get_json()
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id requis'}), 400

    try:
        # Vérifier et supprimer utilisateur Firebase Auth
        auth.get_user(user_id)
        auth.delete_user(user_id)

        # Supprimer documents Firestore
        for collection in ['users', 'userLogs', 'admins']:
            doc_ref = db.collection(collection).document(user_id)
            if doc_ref.get().exists:
                doc_ref.delete()

        return jsonify({'message': f'Utilisateur {user_id} supprimé avec succès'}), 200

    except auth.UserNotFoundError:
        return jsonify({'error': f'Utilisateur {user_id} non trouvé'}), 404

    except Exception as e:
        return jsonify({'error': f'Erreur lors de la suppression: {str(e)}'}), 500


if __name__ == '__main__':
    # Rendre accessible sur tous les interfaces et port 8080 (Render)
    app.run(host='0.0.0.0', port=8080)

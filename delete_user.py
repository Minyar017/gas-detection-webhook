from firebase_admin import initialize_app, auth, firestore
from flask import Flask, request, jsonify
import os

# Initialiser Firebase avec des variables d'environnement
initialize_app({
    'credential': {
        'type': 'service_account',
        'project_id': os.getenv('FIREBASE_PROJECT_ID'),
        'private_key_id': os.getenv('FIREBASE_PRIVATE_KEY_ID'),
        'private_key': os.getenv('FIREBASE_PRIVATE_KEY').replace('\\n', '\n'),
        'client_email': os.getenv('FIREBASE_CLIENT_EMAIL'),
        'client_id': os.getenv('FIREBASE_CLIENT_ID'),
        'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
        'token_uri': 'https://oauth2.googleapis.com/token',
        'auth_provider_x509_cert_url': 'https://www.googleapis.com/oauth2/v1/certs',
        'client_x509_cert_url': os.getenv('FIREBASE_CLIENT_CERT_URL'),
        'universe_domain': 'googleapis.com'
    }
})

app = Flask(__name__)

@app.route('/api/delete_user', methods=['POST'])
def delete_user():
    # Vérifier le token d'authentification
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Non autorisé'}), 401
    id_token = auth_header.split('Bearer ')[1]
    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        db = firestore.client()
        admin_doc = db.collection('admins').document(uid).get()
        if not admin_doc.exists or admin_doc.to_dict().get('role') != 'admin':
            return jsonify({'error': 'Réservé aux admins'}), 403
    except Exception as e:
        return jsonify({'error': f'Erreur auth: {str(e)}'}), 401

    # Obtenir l'user_id
    user_id = request.json.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id requis'}), 400

    # Supprimer l'utilisateur et ses données
    try:
        auth.delete_user(user_id)
        for collection in ['users', 'userLogs', 'admins']:
            db.collection(collection).document(user_id).delete()
        return jsonify({'message': f'Utilisateur {user_id} supprimé'}), 200
    except Exception as e:
        return jsonify({'error': f'Erreur: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8080)))

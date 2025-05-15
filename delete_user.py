from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import auth, firestore

app = Flask(__name__)
CORS(app)  # Active CORS pour toutes les routes et origines

firebase_admin.initialize_app()

@app.route('/delete_user_endpoint', methods=['POST'])
def delete_user():
    # Vérification du token Authorization
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Non autorisé'}), 401

    id_token = auth_header.split('Bearer ')[1]

    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']

        # Vérifier rôle admin (exemple)
        db = firestore.client()
        admin_doc = db.collection('admins').document(uid).get()
        if not admin_doc.exists or admin_doc.to_dict().get('role') != 'admin':
            return jsonify({'error': 'Accès réservé aux admins'}), 403

    except Exception as e:
        return jsonify({'error': f'Erreur d\'authentification: {str(e)}'}), 401

    user_id = request.json.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id requis'}), 400

    try:
        auth.get_user(user_id)  # Vérifier si user existe
        auth.delete_user(user_id)  # Supprimer user Firebase Auth

        # Supprimer documents Firestore
        for collection in ['users', 'userLogs', 'admins']:
            doc_ref = db.collection(collection).document(user_id)
            if doc_ref.get().exists:
                doc_ref.delete()

        return jsonify({'message': f'Utilisateur {user_id} supprimé avec succès'}), 200

    except auth.UserNotFoundError:
        return jsonify({'error': f'Utilisateur {user_id} non trouvé'}), 404
    except Exception as e:
        return jsonify({'error': f'Erreur: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(debug=True)

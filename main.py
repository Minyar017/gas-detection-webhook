from firebase_functions import https_fn
from firebase_admin import initialize_app, auth, firestore

# Initialiser Firebase Admin
initialize_app()

@https_fn.on_request()
def delete_user_endpoint(req: https_fn.Request) -> https_fn.Response:
    # Log pour déboguer
    print("Requête reçue:", req.json)
    
    # Vérifier l'authentification
    auth_header = req.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        print("Erreur: Token manquant ou invalide")
        return https_fn.Response('Non autorisé', status=401)
    
    id_token = auth_header.split('Bearer ')[1]
    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        print(f"Utilisateur authentifié: {uid}")
        # Vérifier si l'utilisateur est admin
        db = firestore.client()
        admin_doc = db.collection('admins').document(uid).get()
        if not admin_doc.exists or admin_doc.to_dict().get('role') != 'admin':
            print(f"Erreur: {uid} n'est pas admin")
            return https_fn.Response('Accès réservé aux admins', status=403)
    except Exception as e:
        print(f"Erreur d'authentification: {str(e)}")
        return https_fn.Response(f"Erreur d'authentification: {str(e)}", status=401)
    
    user_id = req.json.get('user_id')
    if not user_id:
        print("Erreur: user_id manquant")
        return https_fn.Response('user_id requis', status=400)
    
    try:
        # Vérifier si l'utilisateur existe
        auth.get_user(user_id)
        print(f"Utilisateur trouvé: {user_id}")
        # Supprimer de Firebase Authentication
        auth.delete_user(user_id)
        print(f"Utilisateur {user_id} supprimé de Firebase Auth")
        # Supprimer de Firestore
        for collection in ['users', 'userLogs', 'admins']:
            doc_ref = db.collection(collection).document(user_id)
            if doc_ref.get().exists:
                doc_ref.delete()
                print(f"Document supprimé de la collection {collection} pour {user_id}")
        return https_fn.Response(
            f"Utilisateur {user_id} supprimé avec succès", status=200
        )
    except auth.UserNotFoundError:
        print(f"Utilisateur {user_id} non trouvé")
        return https_fn.Response(f"Utilisateur {user_id} non trouvé", status=404)
    except Exception as e:
        print(f"Erreur lors de la suppression: {str(e)}")
        return https_fn.Response(f"Erreur: {str(e)}", status=500)
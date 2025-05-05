import firebase_admin
from firebase_admin import credentials
import os
import json

# Initialize Firebase once
try:
    firebase_credentials = os.getenv("FIREBASE_CREDENTIALS")
    if firebase_credentials:
        cred_dict = json.loads(firebase_credentials)
        cred = credentials.Certificate(cred_dict)
    else:
        cred = credentials.Certificate('serviceAccountKey.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://projet-fin-d-etude-4632f-default-rtdb.firebaseio.com/'
    })
    print("✅ Firebase initialized")
except Exception as e:
    print(f"❌ Error initializing Firebase: {e}")

from werkzeug.middleware.dispatcher import DispatcherMiddleware
from app import app as app_main
from app1 import app as app_one

application = DispatcherMiddleware(app_main, {
    '/app1': app_one
})
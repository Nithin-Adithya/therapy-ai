import os
import json
import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
import pandas as pd

# Initialize Firebase with your credentials
# Note: You need to replace this with your actual Firebase credentials file path
cred_path = os.environ.get("FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json")

# Check if Firebase has already been initialized
firebase_app = None

def initialize_firebase():
    """Initialize Firebase if not already initialized"""
    global firebase_app
    
    if firebase_app is None:
        try:
            if os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                firebase_app = firebase_admin.initialize_app(cred, {
                    'storageBucket': os.environ.get("FIREBASE_STORAGE_BUCKET", "therapy-ai-bucket.appspot.com")
                })
                print("Firebase initialized successfully")
            else:
                print(f"Firebase credentials file not found at {cred_path}")
                # If in development, try to use a mock in-memory database
                from firebase_admin import db
                firebase_app = firebase_admin.initialize_app(None, {
                    'databaseURL': 'https://therapy-ai-default-rtdb.firebaseio.com/'
                })
                print("Firebase initialized with default app (no credentials)")
        except Exception as e:
            print(f"Error initializing Firebase: {str(e)}")
            return False
    
    return True

def get_firestore_db():
    """Get a Firestore database client"""
    if initialize_firebase():
        return firestore.client()
    return None

def get_storage_bucket():
    """Get a Firebase Storage bucket"""
    if initialize_firebase():
        return storage.bucket()
    return None

# =============================================================================
# Conversation Database Functions (Firestore)
# =============================================================================

def init_database():
    """Initialize Firestore database with default therapeutic responses if needed"""
    db = get_firestore_db()
    if not db:
        print("Failed to initialize Firebase database")
        return False
    
    # Check if we already have therapeutic responses
    responses_ref = db.collection('therapeutic_responses')
    responses = list(responses_ref.limit(1).stream())
    
    # If no responses exist, populate with default responses
    if not responses:
        responses = [
            # Very negative responses
            {"sentiment_category": "very_negative", "response_template": "I can see that you're going through a really difficult time. It's brave of you to share these feelings."},
            {"sentiment_category": "very_negative", "response_template": "I'm sorry to hear you're feeling this way. These emotions are challenging, but you're not alone in facing them."},
            {"sentiment_category": "very_negative", "response_template": "That sounds incredibly hard. Would it help to talk more about what specifically is causing these feelings?"},
            {"sentiment_category": "very_negative", "response_template": "When you're experiencing such intense emotions, it can be overwhelming. Let's take it one step at a time."},
            {"sentiment_category": "very_negative", "response_template": "I'm here to listen. Sometimes just expressing these difficult feelings can be a first step toward managing them."},
            
            # Somewhat negative responses
            {"sentiment_category": "somewhat_negative", "response_template": "It sounds like things have been tough lately. What small thing might bring you a moment of peace today?"},
            {"sentiment_category": "somewhat_negative", "response_template": "I'm noticing some difficult emotions in what you're sharing. What has helped you cope with similar feelings in the past?"},
            {"sentiment_category": "somewhat_negative", "response_template": "Sometimes we can feel stuck in negative feelings. Can we explore what might help shift your perspective?"},
            {"sentiment_category": "somewhat_negative", "response_template": "Those feelings are valid. Would it help to discuss some strategies for managing them when they arise?"},
            {"sentiment_category": "somewhat_negative", "response_template": "I appreciate you sharing these thoughts. What do you think triggered these feelings?"},
            
            # Neutral responses
            {"sentiment_category": "neutral", "response_template": "Tell me more about your situation. I'm here to listen and support you."},
            {"sentiment_category": "neutral", "response_template": "How long have you been feeling this way? Understanding the timeline might help us explore this further."},
            {"sentiment_category": "neutral", "response_template": "I'm curious about what prompted you to reach out today?"},
            {"sentiment_category": "neutral", "response_template": "Sometimes talking through our thoughts can help bring clarity. Is there a specific area you'd like to focus on?"},
            {"sentiment_category": "neutral", "response_template": "Everyone's experience is unique. Could you share more about what this means for you personally?"},
            
            # Somewhat positive responses
            {"sentiment_category": "somewhat_positive", "response_template": "I'm glad to hear there are some positive aspects to your situation. Let's build on those."},
            {"sentiment_category": "somewhat_positive", "response_template": "It sounds like you're making progress. What do you think has contributed to these positive changes?"},
            {"sentiment_category": "somewhat_positive", "response_template": "I notice a bit of optimism in your message. How can we nurture that feeling?"},
            {"sentiment_category": "somewhat_positive", "response_template": "That's a really insightful observation. How does recognizing this make you feel?"},
            {"sentiment_category": "somewhat_positive", "response_template": "It seems like you're developing some helpful perspective. What else might support your continued growth?"},
            
            # Very positive responses
            {"sentiment_category": "very_positive", "response_template": "It's wonderful to hear you're feeling so positive! What's contributed most to this uplifted state?"},
            {"sentiment_category": "very_positive", "response_template": "Your positivity is inspiring. How can you use this energy to support your continued well-being?"},
            {"sentiment_category": "very_positive", "response_template": "These positive feelings are something to celebrate. How might you sustain this momentum?"},
            {"sentiment_category": "very_positive", "response_template": "I'm really happy to hear things are going well. What personal strengths have helped you reach this point?"},
            {"sentiment_category": "very_positive", "response_template": "That's excellent progress! What lessons from this success might help you in other areas of your life?"}
        ]
        
        # Add each response to Firestore
        for response in responses:
            responses_ref.add(response)
        print(f"Added {len(responses)} default therapeutic responses to Firestore")
    
    return True

def save_conversation(user_message, ai_response, sentiment_score):
    """Save conversation to Firestore database"""
    db = get_firestore_db()
    if not db:
        print("Failed to access Firestore database")
        return False
    
    timestamp = datetime.datetime.now().isoformat()
    
    # Add conversation to Firestore
    conversations_ref = db.collection('conversations')
    conversations_ref.add({
        'timestamp': timestamp,
        'user_message': user_message,
        'ai_response': ai_response,
        'sentiment_score': sentiment_score
    })
    
    return True

def get_therapeutic_response(sentiment_score):
    """Get a therapeutic response based on sentiment from Firestore"""
    from main import get_sentiment_category
    sentiment_category = get_sentiment_category(sentiment_score)
    
    db = get_firestore_db()
    if not db:
        # Fallback response if database is not available
        return "I'm here to listen and support you. Please tell me more about how you're feeling."
    
    # Query responses for the given sentiment category
    responses_ref = db.collection('therapeutic_responses')
    query = responses_ref.where('sentiment_category', '==', sentiment_category)
    
    # Get all matching responses
    responses = list(query.stream())
    
    if responses:
        # Choose a random response from matching category
        import random
        response_doc = random.choice(responses)
        return response_doc.to_dict()['response_template']
    else:
        # Fallback response
        return "I'm here to listen and support you. Please tell me more about how you're feeling."

# =============================================================================
# Dataset Storage Functions (Firebase Storage)
# =============================================================================

def upload_dataset(file_path, destination_path=None):
    """Upload a dataset file to Firebase Storage"""
    bucket = get_storage_bucket()
    if not bucket:
        print("Failed to access Firebase Storage bucket")
        return False
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    try:
        # If destination path is not provided, use the file name
        if destination_path is None:
            destination_path = os.path.basename(file_path)
        
        # Upload file to Firebase Storage
        blob = bucket.blob(f"datasets/{destination_path}")
        blob.upload_from_filename(file_path)
        
        print(f"Uploaded {file_path} to Firebase Storage as {destination_path}")
        return True
    except Exception as e:
        print(f"Error uploading {file_path} to Firebase Storage: {str(e)}")
        return False

def download_dataset(file_name, destination_path=None):
    """Download a dataset file from Firebase Storage"""
    bucket = get_storage_bucket()
    if not bucket:
        print("Failed to access Firebase Storage bucket")
        return False
    
    try:
        # If destination path is not provided, use the file name
        if destination_path is None:
            destination_path = file_name
        
        # Download file from Firebase Storage
        blob = bucket.blob(f"datasets/{file_name}")
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        blob.download_to_filename(destination_path)
        
        print(f"Downloaded {file_name} from Firebase Storage to {destination_path}")
        return True
    except Exception as e:
        print(f"Error downloading {file_name} from Firebase Storage: {str(e)}")
        return False

# =============================================================================
# Sentiment Model Functions (Firebase Storage)
# =============================================================================

def save_sentiment_model(model, vectorizer, model_name="sentiment_model"):
    """Save trained sentiment model to Firebase Storage"""
    import pickle
    import tempfile
    
    bucket = get_storage_bucket()
    if not bucket:
        print("Failed to access Firebase Storage bucket")
        return False
    
    try:
        # Create temporary files for model and vectorizer
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as model_file:
            pickle.dump(model, model_file)
            model_path = model_file.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)
            vectorizer_path = vectorizer_file.name
        
        # Upload files to Firebase Storage
        model_blob = bucket.blob(f"models/{model_name}_model.pkl")
        model_blob.upload_from_filename(model_path)
        
        vectorizer_blob = bucket.blob(f"models/{model_name}_vectorizer.pkl")
        vectorizer_blob.upload_from_filename(vectorizer_path)
        
        # Clean up temporary files
        os.unlink(model_path)
        os.unlink(vectorizer_path)
        
        print(f"Saved sentiment model to Firebase Storage as {model_name}")
        return True
    except Exception as e:
        print(f"Error saving sentiment model to Firebase Storage: {str(e)}")
        return False

def load_sentiment_model(model_name="sentiment_model"):
    """Load trained sentiment model from Firebase Storage"""
    import pickle
    import tempfile
    
    bucket = get_storage_bucket()
    if not bucket:
        print("Failed to access Firebase Storage bucket")
        return None, None
    
    try:
        # Create temporary files for model and vectorizer
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as model_file:
            model_path = model_file.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as vectorizer_file:
            vectorizer_path = vectorizer_file.name
        
        # Download files from Firebase Storage
        model_blob = bucket.blob(f"models/{model_name}_model.pkl")
        model_blob.download_to_filename(model_path)
        
        vectorizer_blob = bucket.blob(f"models/{model_name}_vectorizer.pkl")
        vectorizer_blob.download_to_filename(vectorizer_path)
        
        # Load model and vectorizer from temporary files
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Clean up temporary files
        os.unlink(model_path)
        os.unlink(vectorizer_path)
        
        print(f"Loaded sentiment model from Firebase Storage: {model_name}")
        return model, vectorizer
    except Exception as e:
        print(f"Error loading sentiment model from Firebase Storage: {str(e)}")
        return None, None 
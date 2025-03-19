import os
import json
import datetime
import random
import requests
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# Import Firebase functions
from firebase_db import init_database, save_conversation, get_therapeutic_response
from firebase_db import upload_dataset, download_dataset, save_sentiment_model, load_sentiment_model
import google.generativeai as genai
import uuid

# Load environment variables
load_dotenv()

# Constants
APP_TITLE = "Therapy AI"
APP_DESCRIPTION = "A therapeutic chatbot with sentiment analysis powered by Gemini AI"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
TWITTER_DATA_PATH = "twitter_training.csv"
REVIEWS_DATA_PATH = "Reviews.csv"

def initialize_app():
    """Initialize the application and database connections"""
    # Initialize Firebase
    init_database()
    
    # Initialize Gemini API
    try:
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.error(f"Error initializing Gemini API: {str(e)}")
    
    # Initialize session state variables if not already set
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'chat_id' not in st.session_state:
        st.session_state.chat_id = str(uuid.uuid4())
    
    if 'sentiment_model' not in st.session_state:
        st.session_state.sentiment_model = None
        
    if 'vectorizer' not in st.session_state:
        st.session_state.vectorizer = None

def download_nltk_resources():
    """Download necessary NLTK resources if not already present"""
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
    except Exception as e:
        st.warning(f"Error downloading NLTK resources: {str(e)}")

def preprocess_text(text):
    """Preprocess text for sentiment analysis"""
    try:
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    except Exception as e:
        print(f"Error preprocessing text: {str(e)}")
        return text  # Return original text on error

def train_sentiment_model():
    """Train a sentiment analysis model on multiple datasets and store in Firebase"""
    if st.session_state.sentiment_model is not None:
        return st.session_state.sentiment_model, st.session_state.vectorizer
    
    # Try to load pre-trained model from Firebase
    model, vectorizer = load_sentiment_model()
    if model is not None and vectorizer is not None:
        # Store model in session state
        st.session_state.sentiment_model = model
        st.session_state.vectorizer = vectorizer
        return model, vectorizer
    
    # Initialize a combined dataframe
    combined_df = None
    
    try:
        # Check if datasets exist locally, if not, download from Firebase
        if not os.path.exists(TWITTER_DATA_PATH):
            download_dataset("twitter_training.csv", TWITTER_DATA_PATH)
        
        if not os.path.exists(REVIEWS_DATA_PATH):
            download_dataset("Reviews.csv", REVIEWS_DATA_PATH)
        
        # Process Twitter data with smaller sample limit for faster loading
        twitter_df = process_twitter_dataset(display_info=False, sample_limit=5000)
        
        # Process Reviews data with smaller sample limit for faster loading
        reviews_df = process_reviews_dataset(display_info=False, sample_limit=5000)
        
        # Combine datasets if both are available
        if twitter_df is not None and reviews_df is not None:
            combined_df = pd.concat([twitter_df, reviews_df], ignore_index=True)
        elif twitter_df is not None:
            combined_df = twitter_df
        elif reviews_df is not None:
            combined_df = reviews_df
        else:
            return None, None
        
        # Use fewer features for faster training
        vectorizer = TfidfVectorizer(max_features=2000, 
                                     min_df=5,
                                     max_df=0.8,
                                     sublinear_tf=True)
        X = vectorizer.fit_transform(combined_df['ProcessedText'])
        y = combined_df['SentimentScore']
        
        # Split data with smaller test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        
        # Train model with faster settings
        model = LogisticRegression(max_iter=300, solver='saga', n_jobs=-1, C=1.0)
        model.fit(X_train, y_train)
        
        # Store model and vectorizer in session state
        st.session_state.sentiment_model = model
        st.session_state.vectorizer = vectorizer
        
        # Save model to Firebase
        save_sentiment_model(model, vectorizer)
        
        return model, vectorizer
    except Exception as e:
        print(f"Error training sentiment model: {str(e)}")
        return None, None

def process_twitter_dataset(display_info=True, sample_limit=None):
    """Process the Twitter dataset for sentiment analysis"""
    try:
        # Check if dataset exists locally, if not, download from Firebase
        if not os.path.exists(TWITTER_DATA_PATH):
            download_dataset("twitter_training.csv", TWITTER_DATA_PATH)
            
        # Check if file exists after attempted download
        if not os.path.exists(TWITTER_DATA_PATH):
            if display_info:
                st.error("Twitter dataset not found and could not be downloaded from Firebase.")
            return None
            
        # Load the Twitter dataset
        encodings = ['utf-8', 'latin1', 'ISO-8859-1']
        for encoding in encodings:
            try:
                twitter_df = pd.read_csv(TWITTER_DATA_PATH, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            if display_info:
                st.error("Failed to decode Twitter dataset with available encodings.")
            return None
        
        # Sample data if requested (for faster processing)
        if sample_limit and len(twitter_df) > sample_limit:
            twitter_df = twitter_df.sample(n=sample_limit, random_state=42)
        
        # Map sentiment to numerical values (0 = negative, 1 = neutral, 2 = positive)
        sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        twitter_df['SentimentScore'] = twitter_df['sentiment'].map(sentiment_map)
        
        # Text preprocessing
        twitter_df['ProcessedText'] = twitter_df['text'].astype(str).apply(preprocess_text)
        
        if display_info:
            st.info(f"Twitter dataset processed: {len(twitter_df)} rows")
            st.write("Sentiment distribution:")
            sentiment_counts = twitter_df['sentiment'].value_counts()
            st.write(sentiment_counts)
            
        return twitter_df[['ProcessedText', 'SentimentScore']]
    except Exception as e:
        if display_info:
            st.error(f"Error processing Twitter dataset: {str(e)}")
        print(f"Error processing Twitter dataset: {str(e)}")
        return None

def process_reviews_dataset(display_info=True, sample_limit=None):
    """Process the Reviews dataset for sentiment analysis"""
    try:
        # Check if dataset exists locally, if not, download from Firebase
        if not os.path.exists(REVIEWS_DATA_PATH):
            download_dataset("Reviews.csv", REVIEWS_DATA_PATH)
            
        # Check if file exists after attempted download
        if not os.path.exists(REVIEWS_DATA_PATH):
            if display_info:
                st.error("Reviews dataset not found and could not be downloaded from Firebase.")
            return None
            
        # Load the Reviews dataset
        encodings = ['utf-8', 'latin1', 'ISO-8859-1']
        for encoding in encodings:
            try:
                reviews_df = pd.read_csv(REVIEWS_DATA_PATH, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            if display_info:
                st.error("Failed to decode Reviews dataset with available encodings.")
            return None
        
        # Check if 'Score' column exists
        if 'Score' not in reviews_df.columns:
            if display_info:
                st.error("Reviews dataset does not contain expected 'Score' column.")
            return None
        
        # Sample data if requested (for faster processing)
        if sample_limit and len(reviews_df) > sample_limit:
            reviews_df = reviews_df.sample(n=sample_limit, random_state=42)
        
        # Map ratings to sentiment (1-2 = negative, 3 = neutral, 4-5 = positive)
        def map_score_to_sentiment(score):
            if score <= 2:
                return 0  # negative
            elif score == 3:
                return 1  # neutral
            else:
                return 2  # positive
        
        reviews_df['SentimentScore'] = reviews_df['Score'].apply(map_score_to_sentiment)
        
        # Text preprocessing - combine Text and Summary if both exist
        if 'Text' in reviews_df.columns and 'Summary' in reviews_df.columns:
            reviews_df['CombinedText'] = reviews_df['Summary'].astype(str) + " " + reviews_df['Text'].astype(str)
        elif 'Text' in reviews_df.columns:
            reviews_df['CombinedText'] = reviews_df['Text'].astype(str)
        elif 'Summary' in reviews_df.columns:
            reviews_df['CombinedText'] = reviews_df['Summary'].astype(str)
        else:
            if display_info:
                st.error("Reviews dataset does not contain expected text columns.")
            return None
        
        reviews_df['ProcessedText'] = reviews_df['CombinedText'].apply(preprocess_text)
        
        if display_info:
            st.info(f"Reviews dataset processed: {len(reviews_df)} rows")
            st.write("Sentiment distribution:")
            sentiment_counts = reviews_df['SentimentScore'].value_counts()
            st.write(sentiment_counts)
            
        return reviews_df[['ProcessedText', 'SentimentScore']]
    except Exception as e:
        if display_info:
            st.error(f"Error processing Reviews dataset: {str(e)}")
        print(f"Error processing Reviews dataset: {str(e)}")
        return None

def analyze_sentiment(text):
    """Analyze sentiment of text using trained model or get from Firebase if not trained yet"""
    try:
        # Get or train sentiment model
        model, vectorizer = train_sentiment_model()
        
        if model is None or vectorizer is None:
            return 'neutral', 0.5  # Default if model not available
        
        # Preprocess the text
        processed_text = preprocess_text(text)
        
        # Transform text using vectorizer
        text_vectorized = vectorizer.transform([processed_text])
        
        # Predict sentiment
        sentiment_score = model.predict(text_vectorized)[0]
        
        # Map numerical score to text sentiment
        if sentiment_score == 0:
            sentiment = 'negative'
            score_normalized = 0.25  # Normalized score for visualization
        elif sentiment_score == 1:
            sentiment = 'neutral'
            score_normalized = 0.5
        else:  # sentiment_score == 2
            sentiment = 'positive'
            score_normalized = 0.75
            
        return sentiment, score_normalized
    except Exception as e:
        print(f"Error analyzing sentiment: {str(e)}")
        return 'neutral', 0.5  # Default fallback

def generate_response(prompt, chat_history):
    """Generate response using Gemini or get therapeutic response from Firebase based on sentiment"""
    try:
        # Analyze sentiment of user's message
        sentiment, _ = analyze_sentiment(prompt)
        
        # Try to get a therapeutic response from Firebase based on sentiment
        therapeutic_response = get_therapeutic_response(sentiment)
        
        # If we have a therapeutic response and it's appropriate (20% chance), use it
        if therapeutic_response and random.random() < 0.2:
            return therapeutic_response
        
        # Otherwise use Gemini API
        if GEMINI_API_KEY:
            # Format the chat history for the model
            formatted_history = []
            for message in chat_history:
                role = "user" if message["role"] == "user" else "model"
                formatted_history.append({"role": role, "parts": [message["content"]]})
            
            # Add the current prompt
            formatted_history.append({"role": "user", "parts": [prompt]})
            
            # Set up the model with custom parameters
            model = genai.GenerativeModel(
                model_name="gemini-pro",
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 800,
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ],
            )
            
            # Generate response
            response = model.generate_content(formatted_history)
            return response.text
        else:
            # Fallback to therapeutic responses if API key is not available
            if therapeutic_response:
                return therapeutic_response
            else:
                return "I'm here to listen and help. Could you tell me more about what you're feeling?"
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "I'm having trouble processing that. Let's take a step back. How are you feeling right now?"

def handle_chat_input():
    """Handle chat input from the user"""
    # Get user input
    user_input = st.chat_input("Type your message here...")
    
    # Process user input
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Generate AI response
        with st.spinner("Thinking..."):
            response = generate_response(user_input, st.session_state.messages)
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Save conversation to Firebase
        save_conversation(
            st.session_state.chat_id,
            st.session_state.messages[-2]["content"],  # User message
            st.session_state.messages[-1]["content"],  # AI response
            analyze_sentiment(user_input)[0]  # Sentiment of user message
        )

def display_chat():
    """Display the chat interface"""
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Handle user input
    handle_chat_input()

def is_mobile_device():
    """Detect if user is on a mobile device using user-agent"""
    try:
        user_agent = st.get_complex_state()['request']['headers']['User-Agent']
        mobile_patterns = ['Android', 'webOS', 'iPhone', 'iPad', 'iPod', 'BlackBerry', 'IEMobile', 'Opera Mini']
        return any(pattern in user_agent for pattern in mobile_patterns)
    except:
        # If we can't detect, default to desktop
        return False

def main():
    """Main function to run the application"""
    # Set page config
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="💭",
        layout="wide",
        initial_sidebar_state="expanded" if not is_mobile_device() else "collapsed"
    )
    
    # Initialize app and database
    initialize_app()
    
    # Add CSS to hide sidebar on mobile
    st.markdown("""
    <style>
    @media (max-width: 768px) {
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        button[kind="header"] {
            display: none !important;
        }
        .main .block-container {
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 2rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add JavaScript to force disable sidebar on mobile
    st.markdown("""
    <script>
    function hideSidebarOnMobile() {
        if (/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)) {
            // Hide sidebar
            const sidebar = document.querySelector('section[data-testid="stSidebar"]');
            if (sidebar) sidebar.style.display = 'none';
            
            // Hide toggle button
            const toggleButton = document.querySelector('button[kind="header"]');
            if (toggleButton) toggleButton.style.display = 'none';
        }
    }
    
    // Run on page load and also after a short delay to ensure elements are loaded
    if (document.readyState === 'complete') {
        hideSidebarOnMobile();
    } else {
        window.addEventListener('load', hideSidebarOnMobile);
    }
    
    // Also try after a slight delay to catch any late-loaded elements
    setTimeout(hideSidebarOnMobile, 500);
    </script>
    """, unsafe_allow_html=True)
    
    # Set app title and description
    st.title(APP_TITLE)
    st.write(APP_DESCRIPTION)
    
    # Create sidebar for sentiment visualization (only if not on mobile)
    if not is_mobile_device():
        with st.sidebar:
            st.subheader("Sentiment Analysis")
            st.write("This shows the sentiment of your messages over time.")
            
            if st.button("Train Sentiment Model"):
                with st.spinner("Training model..."):
                    model, vectorizer = train_sentiment_model()
                    if model is not None:
                        st.success("Model trained successfully!")
                    else:
                        st.error("Failed to train model.")
            
            # Option to upload datasets to Firebase
            st.subheader("Dataset Management")
            
            upload_twitter = st.button("Upload Twitter Dataset to Firebase")
            if upload_twitter:
                with st.spinner("Uploading Twitter dataset..."):
                    if os.path.exists(TWITTER_DATA_PATH):
                        success = upload_dataset(TWITTER_DATA_PATH, "twitter_training.csv")
                        if success:
                            st.success("Twitter dataset uploaded successfully!")
                        else:
                            st.error("Failed to upload Twitter dataset.")
                    else:
                        st.error(f"Twitter dataset not found at {TWITTER_DATA_PATH}")
            
            upload_reviews = st.button("Upload Reviews Dataset to Firebase")
            if upload_reviews:
                with st.spinner("Uploading Reviews dataset..."):
                    if os.path.exists(REVIEWS_DATA_PATH):
                        success = upload_dataset(REVIEWS_DATA_PATH, "Reviews.csv")
                        if success:
                            st.success("Reviews dataset uploaded successfully!")
                        else:
                            st.error("Failed to upload Reviews dataset.")
                    else:
                        st.error(f"Reviews dataset not found at {REVIEWS_DATA_PATH}")
    
    # Display chat interface
    display_chat()

if __name__ == "__main__":
    main() 
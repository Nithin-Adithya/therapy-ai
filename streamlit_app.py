import os
import json
import sqlite3
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

# Load environment variables
load_dotenv()

# Constants
APP_TITLE = "Therapy AI"
APP_DESCRIPTION = "A therapeutic chatbot with sentiment analysis powered by Gemini AI"
DB_PATH = "therapy_data.db"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
TWITTER_DATA_PATH = "twitter_training.csv"
REVIEWS_DATA_PATH = "Reviews.csv"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "sentiment_history" not in st.session_state:
    st.session_state.sentiment_history = []

if "sentiment_model" not in st.session_state:
    st.session_state.sentiment_model = None

if "vectorizer" not in st.session_state:
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
    """Preprocess text for sentiment analysis with optimized performance"""
    try:
        # Convert to lowercase and ensure string type
        text = str(text).lower()
        
        # Combine regex operations to reduce passes
        text = re.sub(r'http\S+|www\S+|https\S+|@\w+|\#\w+|[^a-zA-Z\s]', ' ', text)
        
        # Split by whitespace (faster than formal tokenization)
        tokens = text.split()
        
        # Use set lookup for stopwords (O(1) complexity)
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        except:
            pass
        
        # Optional: lemmatize only if needed for accuracy
        # This is expensive, so consider skipping for performance
        if len(tokens) > 0:
            try:
                lemmatizer = WordNetLemmatizer()
                # Process in batches
                BATCH_SIZE = 1000
                lemmatized_tokens = []
                for i in range(0, len(tokens), BATCH_SIZE):
                    batch = tokens[i:i+BATCH_SIZE]
                    lemmatized_tokens.extend([lemmatizer.lemmatize(word) for word in batch])
                tokens = lemmatized_tokens
            except:
                pass
        
        # Join tokens back into text
        text = ' '.join(tokens)
        
        return text
    except Exception as e:
        # Return original text if preprocessing fails
        print(f"Text preprocessing error: {str(e)}. Using original text.")
        return str(text)

def train_sentiment_model():
    """Train a sentiment analysis model on multiple datasets"""
    if st.session_state.sentiment_model is not None:
        return st.session_state.sentiment_model, st.session_state.vectorizer
    
    # Initialize a combined dataframe
    combined_df = None
    
    try:
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
        
        return model, vectorizer
    except Exception as e:
        print(f"Error training sentiment model: {str(e)}")
        return None, None

def process_twitter_dataset(display_info=False, sample_limit=None):
    """Process Twitter dataset for sentiment analysis with optimized loading"""
    try:
        # Define possible file paths
        file_paths = [
            TWITTER_DATA_PATH,
            os.path.join('sample_data', 'twitter_sample.csv')
        ]
        
        # Try each path
        twitter_path = None
        for path in file_paths:
            if os.path.exists(path):
                twitter_path = path
                break
                
        if twitter_path is None:
            if display_info:
                st.warning(f"Twitter dataset file not found in any of the expected locations.")
            return None
            
        # Quick approach for very fast loading with head() when sample is small
        if sample_limit and sample_limit <= 5000:
            try:
                # Try simple head() approach first (much faster)
                df = pd.read_csv(twitter_path, header=None if twitter_path.endswith('twitter_training.csv') else 0, 
                               encoding='utf-8', nrows=sample_limit, low_memory=False)
            except:
                try:
                    df = pd.read_csv(twitter_path, header=None if twitter_path.endswith('twitter_training.csv') else 0, 
                                   encoding='latin1', nrows=sample_limit, low_memory=False)
                except Exception as e:
                    if display_info:
                        st.error(f"Failed to load Twitter dataset: {str(e)}")
                    return None
        else:
            # Original sampling approach for larger samples
            try:
                if sample_limit:
                    # Count lines for sampling
                    with open(twitter_path, 'rb') as f:
                        approx_lines = sum(1 for _ in f) - 1  # Exclude header
                    
                    # Calculate skiprows for random sampling
                    if approx_lines > sample_limit:
                        skiprows = sorted(random.sample(range(1, approx_lines + 1), 
                                                      approx_lines - sample_limit))
                        df = pd.read_csv(twitter_path, header=None if twitter_path.endswith('twitter_training.csv') else 0, 
                                       encoding='utf-8', skiprows=skiprows, low_memory=False)
                    else:
                        df = pd.read_csv(twitter_path, header=None if twitter_path.endswith('twitter_training.csv') else 0, 
                                       encoding='utf-8', low_memory=False)
                else:
                    df = pd.read_csv(twitter_path, header=None if twitter_path.endswith('twitter_training.csv') else 0, 
                                   encoding='utf-8', low_memory=False)
            except:
                try:
                    if sample_limit:
                        # Count lines for sampling
                        with open(twitter_path, 'rb') as f:
                            approx_lines = sum(1 for _ in f) - 1  # Exclude header
                        
                        # Calculate skiprows for random sampling
                        if approx_lines > sample_limit:
                            skiprows = sorted(random.sample(range(1, approx_lines + 1), 
                                                          approx_lines - sample_limit))
                            df = pd.read_csv(twitter_path, header=None if twitter_path.endswith('twitter_training.csv') else 0, 
                                           encoding='latin1', skiprows=skiprows, low_memory=False)
                        else:
                            df = pd.read_csv(twitter_path, header=None if twitter_path.endswith('twitter_training.csv') else 0, 
                                           encoding='latin1', low_memory=False)
                    else:
                        df = pd.read_csv(twitter_path, header=None if twitter_path.endswith('twitter_training.csv') else 0, 
                                       encoding='latin1', low_memory=False)
                except Exception as e:
                    if display_info:
                        st.error(f"Failed to load Twitter dataset: {str(e)}")
                    return None
        
        # Display dataset info if needed
        if display_info:
            st.info(f"Loaded Twitter dataset from {twitter_path} with {len(df)} rows")
        
        # Handle different column names based on the file used
        if twitter_path.endswith('twitter_sample.csv'):
            # Sample file already has headers
            if 'ID' in df.columns and 'Sentiment' in df.columns and 'Text' in df.columns:
                # Rename to match expected format
                df.rename(columns={'ID': 'ID'}, inplace=True)
            else:
                if display_info:
                    st.warning("Twitter sample dataset doesn't have the expected format.")
                return None
        else:
            # Original dataset needs column names
            # Assign column names based on number of columns
            if len(df.columns) >= 4:
                df.columns = ['ID', 'Topic', 'Sentiment', 'Text'] + [f'Extra_{i}' for i in range(len(df.columns) - 4)]
            elif len(df.columns) == 3:
                df.columns = ['ID', 'Sentiment', 'Text']
            else:
                if display_info:
                    st.warning("Twitter dataset doesn't have the expected format.")
                return None
        
        # Map sentiments to numerical values
        sentiment_map = {
            'Positive': 1,
            'Negative': -1,
            'Neutral': 0
        }
        
        # Convert sentiment to numerical values
        df['SentimentScore'] = df['Sentiment'].map(sentiment_map)
        
        # Check if we have valid sentiment scores
        if df['SentimentScore'].isna().all():
            if display_info:
                st.warning("Could not map sentiment values in Twitter dataset.")
            return None
        
        # Drop rows with NaN sentiment scores
        df = df.dropna(subset=['SentimentScore'])
        
        if len(df) == 0:
            if display_info:
                st.warning("No valid sentiment data available in Twitter dataset.")
            return None
        
        # Simple preprocess for faster initial load
        processed_texts = []
        for text in df['Text']:
            try:
                # Simplified preprocessing for speed
                text = str(text).lower()
                text = re.sub(r'[^a-zA-Z\s]', ' ', text)
                text = ' '.join([w for w in text.split() if len(w) > 2])
                processed_texts.append(text)
            except:
                processed_texts.append("")
                
        df['ProcessedText'] = processed_texts
        
        return df
    except Exception as e:
        if display_info:
            st.error(f"Error processing Twitter dataset: {str(e)}")
        return None

def process_reviews_dataset(display_info=False, sample_limit=None):
    """Process Reviews dataset for sentiment analysis with optimized processing"""
    try:
        # Define possible file paths
        file_paths = [
            REVIEWS_DATA_PATH,
            os.path.join('sample_data', 'Reviews_sample.csv')
        ]
        
        # Try each path
        reviews_path = None
        for path in file_paths:
            if os.path.exists(path):
                reviews_path = path
                break
                
        if reviews_path is None:
            if display_info:
                st.warning(f"Reviews dataset file not found in any of the expected locations.")
            return None
        
        # Quick approach for very fast loading with head() when sample is small
        if sample_limit and sample_limit <= 5000:
            try:
                # Try simple head() approach first (much faster)
                df = pd.read_csv(reviews_path, encoding='utf-8', nrows=sample_limit)
            except:
                try:
                    df = pd.read_csv(reviews_path, encoding='latin1', nrows=sample_limit)
                except Exception as e:
                    if display_info:
                        st.error(f"Failed to load Reviews dataset: {str(e)}")
                    return None
        else:
            # Original sampling approach for larger samples
            try:
                if sample_limit:
                    # Count lines for sampling
                    with open(reviews_path, 'rb') as f:
                        approx_lines = sum(1 for _ in f) - 1  # Exclude header
                    
                    # Calculate skiprows for random sampling
                    if approx_lines > sample_limit:
                        skiprows = sorted(random.sample(range(1, approx_lines + 1), 
                                                      approx_lines - sample_limit))
                        df = pd.read_csv(reviews_path, encoding='utf-8', skiprows=skiprows)
                    else:
                        df = pd.read_csv(reviews_path, encoding='utf-8')
                else:
                    df = pd.read_csv(reviews_path, encoding='utf-8')
            except:
                try:
                    if sample_limit:
                        # Count lines for sampling
                        with open(reviews_path, 'rb') as f:
                            approx_lines = sum(1 for _ in f) - 1  # Exclude header
                        
                        # Calculate skiprows for random sampling
                        if approx_lines > sample_limit:
                            skiprows = sorted(random.sample(range(1, approx_lines + 1), 
                                                          approx_lines - sample_limit))
                            df = pd.read_csv(reviews_path, encoding='latin1', skiprows=skiprows)
                        else:
                            df = pd.read_csv(reviews_path, encoding='latin1')
                    else:
                        df = pd.read_csv(reviews_path, encoding='latin1')
                except Exception as e:
                    if display_info:
                        st.error(f"Failed to load Reviews dataset: {str(e)}")
                    return None
        
        # Display dataset info if needed
        if display_info:
            st.info(f"Loaded Reviews dataset from {reviews_path} with {len(df)} rows")
        
        # Check if the dataset has the required columns
        if 'Score' not in df.columns or 'Text' not in df.columns:
            if display_info:
                st.warning("Reviews dataset doesn't have the required columns (Score, Text).")
            return None
        
        # Map scores to sentiment categories
        df['Sentiment'] = df['Score'].apply(lambda x: 'Positive' if x > 3 else ('Neutral' if x == 3 else 'Negative'))
        
        # Map sentiments to numerical values for model training
        sentiment_map = {
            'Positive': 1,
            'Negative': -1,
            'Neutral': 0
        }
        
        # Convert sentiment to numerical values
        df['SentimentScore'] = df['Sentiment'].map(sentiment_map)
        
        # Simple preprocess for faster initial load
        processed_texts = []
        for text in df['Text']:
            try:
                # Simplified preprocessing for speed
                text = str(text).lower()
                text = re.sub(r'[^a-zA-Z\s]', ' ', text)
                text = ' '.join([w for w in text.split() if len(w) > 2])
                processed_texts.append(text)
            except:
                processed_texts.append("")
        
        df['ProcessedText'] = processed_texts
        
        return df
    except Exception as e:
        if display_info:
            st.error(f"Error processing Reviews dataset: {str(e)}")
        return None

def analyze_sentiment_with_model(text, model=None, vectorizer=None):
    """Analyze sentiment using trained model or fallback to TextBlob"""
    if model is not None and vectorizer is not None:
        try:
            # Preprocess the text
            processed_text = preprocess_text(text)
            
            # Transform text using vectorizer
            text_vec = vectorizer.transform([processed_text])
            
            # Predict sentiment
            sentiment_score = model.predict(text_vec)[0]
            
            return sentiment_score
        except Exception as e:
            st.warning(f"Custom model error: {e}. Falling back to TextBlob.")
            # Fallback to TextBlob
            return TextBlob(text).sentiment.polarity
    else:
        # Use TextBlob if no model is available
        return TextBlob(text).sentiment.polarity

def init_database():
    """Initialize SQLite database for storing conversation and sentiment data"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        user_message TEXT,
        ai_response TEXT,
        sentiment_score REAL
    )
    ''')
    
    # Create table for therapeutic responses
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS therapeutic_responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sentiment_category TEXT,
        response_template TEXT
    )
    ''')
    
    # Check if we already have therapeutic responses
    cursor.execute("SELECT COUNT(*) FROM therapeutic_responses")
    count = cursor.fetchone()[0]
    
    # If no responses exist, populate with default responses
    if count == 0:
        responses = [
            # Very negative responses
            ("very_negative", "I can see that you're going through a really difficult time. It's brave of you to share these feelings."),
            ("very_negative", "I'm sorry to hear you're feeling this way. These emotions are challenging, but you're not alone in facing them."),
            ("very_negative", "That sounds incredibly hard. Would it help to talk more about what specifically is causing these feelings?"),
            ("very_negative", "When you're experiencing such intense emotions, it can be overwhelming. Let's take it one step at a time."),
            ("very_negative", "I'm here to listen. Sometimes just expressing these difficult feelings can be a first step toward managing them."),
            
            # Somewhat negative responses
            ("somewhat_negative", "It sounds like things have been tough lately. What small thing might bring you a moment of peace today?"),
            ("somewhat_negative", "I'm noticing some difficult emotions in what you're sharing. What has helped you cope with similar feelings in the past?"),
            ("somewhat_negative", "Sometimes we can feel stuck in negative feelings. Can we explore what might help shift your perspective?"),
            ("somewhat_negative", "Those feelings are valid. Would it help to discuss some strategies for managing them when they arise?"),
            ("somewhat_negative", "I appreciate you sharing these thoughts. What do you think triggered these feelings?"),
            
            # Neutral responses
            ("neutral", "Tell me more about your situation. I'm here to listen and support you."),
            ("neutral", "How long have you been feeling this way? Understanding the timeline might help us explore this further."),
            ("neutral", "I'm curious about what prompted you to reach out today?"),
            ("neutral", "Sometimes talking through our thoughts can help bring clarity. Is there a specific area you'd like to focus on?"),
            ("neutral", "Everyone's experience is unique. Could you share more about what this means for you personally?"),
            
            # Somewhat positive responses
            ("somewhat_positive", "I'm glad to hear there are some positive aspects to your situation. Let's build on those."),
            ("somewhat_positive", "It sounds like you're making progress. What do you think has contributed to these positive changes?"),
            ("somewhat_positive", "I notice a bit of optimism in your message. How can we nurture that feeling?"),
            ("somewhat_positive", "That's a really insightful observation. How does recognizing this make you feel?"),
            ("somewhat_positive", "It seems like you're developing some helpful perspective. What else might support your continued growth?"),
            
            # Very positive responses
            ("very_positive", "It's wonderful to hear you're feeling so positive! What's contributed most to this uplifted state?"),
            ("very_positive", "Your positivity is inspiring. How can you use this energy to support your continued well-being?"),
            ("very_positive", "These positive feelings are something to celebrate. How might you sustain this momentum?"),
            ("very_positive", "I'm really happy to hear things are going well. What personal strengths have helped you reach this point?"),
            ("very_positive", "That's excellent progress! What lessons from this success might help you in other areas of your life?")
        ]
        
        cursor.executemany("INSERT INTO therapeutic_responses (sentiment_category, response_template) VALUES (?, ?)", responses)
    
    conn.commit()
    conn.close()

def analyze_sentiment(text):
    """Analyze sentiment of text efficiently with caching for repeated phrases"""
    # Check if we've analyzed this exact text before (simple cache)
    if not hasattr(analyze_sentiment, "cache"):
        analyze_sentiment.cache = {}
    
    # Check cache first (for common greetings/phrases)
    if text in analyze_sentiment.cache:
        return analyze_sentiment.cache[text]
    
    model = st.session_state.sentiment_model
    vectorizer = st.session_state.vectorizer
    
    result = analyze_sentiment_with_model(text, model, vectorizer)
    
    # Store in cache if text is relatively short (to avoid memory bloat)
    if len(text) < 100:
        analyze_sentiment.cache[text] = result
    
    return result

def get_sentiment_label(score):
    """Convert sentiment score to human-readable label"""
    if score > 0.3:
        return "Very Positive"
    elif score > 0:
        return "Somewhat Positive"
    elif score == 0:
        return "Neutral"
    elif score > -0.3:
        return "Somewhat Negative"
    else:
        return "Very Negative"

def get_sentiment_category(score):
    """Convert sentiment score to category for database"""
    if score > 0.3:
        return "very_positive"
    elif score > 0:
        return "somewhat_positive"
    elif score == 0:
        return "neutral"
    elif score > -0.3:
        return "somewhat_negative"
    else:
        return "very_negative"

def save_conversation(user_message, ai_response, sentiment_score):
    """Save conversation to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    
    cursor.execute(
        "INSERT INTO conversations (timestamp, user_message, ai_response, sentiment_score) VALUES (?, ?, ?, ?)",
        (timestamp, user_message, ai_response, sentiment_score)
    )
    
    conn.commit()
    conn.close()

def get_therapeutic_response(sentiment_score):
    """Get a therapeutic response based on sentiment as fallback"""
    sentiment_category = get_sentiment_category(sentiment_score)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all responses for the sentiment category
    cursor.execute(
        "SELECT response_template FROM therapeutic_responses WHERE sentiment_category = ?",
        (sentiment_category,)
    )
    
    responses = cursor.fetchall()
    conn.close()
    
    if responses:
        # Choose a random response from matching category
        return random.choice(responses)[0]
    else:
        # Fallback response
        return "I'm here to listen and support you. Please tell me more about how you're feeling."

def format_conversation_for_gemini(messages):
    """Format the conversation history for Gemini API - optimized version"""
    system_message = None
    
    # Extract system message if present
    if messages and messages[0]["role"] == "system":
        system_message = messages[0]["content"]
        messages = messages[1:]
    
    # Use string builder pattern for better performance
    conversation_parts = []
    
    if system_message:
        conversation_parts.append(f"System instructions: {system_message}\n\n")
    
    # Only include the last 5 messages to reduce token count
    message_limit = min(len(messages), 5)
    for msg in messages[-message_limit:]:
        if msg["role"] == "user":
            conversation_parts.append(f"User: {msg['content']}\n")
        elif msg["role"] == "assistant":
            conversation_parts.append(f"Assistant: {msg['content']}\n")
    
    # Add final prompt for the assistant to respond
    conversation_parts.append("Assistant: ")
    
    return ''.join(conversation_parts)

def call_gemini_api(messages):
    """Call Gemini API with the conversation history - optimized version"""
    try:
        # Format conversation for Gemini
        conversation_text = format_conversation_for_gemini(messages)
        
        # Prepare the request payload with more efficient settings
        payload = {
            "contents": [{
                "parts": [{"text": conversation_text}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 300,  # Reduced for faster response
                "topK": 40,
                "topP": 0.95
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add timeout for API call
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        
        # Extract the generated text from the response
        response_json = response.json()
        if 'candidates' in response_json and len(response_json['candidates']) > 0:
            generated_text = response_json['candidates'][0]['content']['parts'][0]['text']
            return generated_text
        else:
            raise ValueError("No response content found in API response")
            
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        # Use local fallback if API fails - more efficient version
        sentiment_score = analyze_sentiment(messages[-1]["content"])
        return get_therapeutic_response(sentiment_score)

def analyze_sentiment_trend(conversation_history):
    """Analyze the trend of sentiment scores in the conversation history"""
    # Get sentiment scores from the last few exchanges
    recent_scores = [msg.get("sentiment", 0) for msg in conversation_history[-6:] if msg["role"] == "user"]
    
    if len(recent_scores) < 2:
        return "stable"
    
    # Calculate the overall trend
    if recent_scores[-1] > recent_scores[0] + 0.1:
        return "improving"
    elif recent_scores[-1] < recent_scores[0] - 0.1:
        return "declining"
    elif max(recent_scores) - min(recent_scores) > 0.3:
        return "fluctuating"
    else:
        return "stable"

def generate_therapeutic_system_prompt(sentiment_score):
    """Generate system prompt for the AI to act as a therapist"""
    sentiment_category = get_sentiment_category(sentiment_score)
    
    base_prompt = """You are Therapy AI, a compassionate and empathetic AI therapist.
Your goal is to provide supportive, thoughtful responses to users who are seeking emotional support or guidance.
- Listen carefully and validate users' feelings
- Ask thoughtful questions to better understand their situation
- Provide gentle guidance without being judgmental
- Suggest practical coping strategies when appropriate
- Maintain a warm, supportive tone throughout the conversation
- Never claim to diagnose medical conditions or replace professional healthcare
- If someone is in crisis, suggest they contact emergency services or crisis hotlines
"""
    
    # Add sentiment-specific guidance
    if sentiment_category == "very_negative":
        base_prompt += """The user appears to be expressing very negative emotions right now.
- Respond with extra empathy and validation
- Acknowledge the difficulty of their situation
- Be especially gentle and supportive
- Focus on immediate coping strategies if appropriate
- Make sure they know they're not alone in these feelings"""
    elif sentiment_category == "somewhat_negative":
        base_prompt += """The user appears to be expressing somewhat negative emotions.
- Validate their feelings while gently exploring potential perspectives
- Help them identify small steps that might improve their situation
- Balance acknowledgment of difficulties with encouragement"""
    elif sentiment_category == "neutral":
        base_prompt += """The user's emotions appear neutral at the moment.
- Ask thoughtful questions to better understand their situation
- Help them explore their thoughts and feelings more deeply
- Provide a balanced perspective in your responses"""
    elif sentiment_category == "somewhat_positive":
        base_prompt += """The user appears to be expressing somewhat positive emotions.
- Reinforce and build upon these positive elements
- Explore how these positive aspects might be expanded
- Help them identify what's working well and why"""
    elif sentiment_category == "very_positive":
        base_prompt += """The user appears to be expressing very positive emotions.
- Celebrate their positive state
- Help them reflect on what contributed to these feelings
- Discuss how they might maintain this positive momentum"""
    
    return {"role": "system", "content": base_prompt}

def main():
    # Configure app FIRST - must be the first Streamlit command
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"  # Default to expanded, we'll collapse on mobile with JS
    )
    
    # Check if on mobile device
    try:
        # Get screen width via client-side detection
        screen_width_code = """
        <script>
        window.addEventListener('load', function() {
            var screenWidth = window.innerWidth;
            document.getElementById('screenWidthElement').textContent = screenWidth;
        });
        </script>
        <div id="screenWidthElement" style="display:none;"></div>
        """
        st.markdown(screen_width_code, unsafe_allow_html=True)
        
        # Default to desktop view (show sidebar)
        is_mobile_view = False
    except:
        # Default to desktop view if detection fails
        is_mobile_view = False
    
    # Add mobile detection and CSS for different devices
    st.markdown("""
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const isMobile = window.innerWidth < 768;
        
        if (isMobile) {
            // For mobile: Hide sidebar completely
            const style = document.createElement('style');
            style.innerHTML = `
                [data-testid="stSidebar"] {display: none !important;}
                [data-testid="collapsedControl"] {display: none !important;}
                .main .block-container {max-width: 100% !important; padding: 1rem !important;}
            `;
            document.head.appendChild(style);
            
            // Try to remove sidebar elements
            setTimeout(() => {
                const sidebar = document.querySelector('[data-testid="stSidebar"]');
                if (sidebar) sidebar.style.display = 'none';
                
                const toggleButton = document.querySelector('[data-testid="collapsedControl"]');
                if (toggleButton) toggleButton.style.display = 'none';
            }, 100);
        }
    });
    </script>
    
    <style>
    /* Base styling */
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    
    /* Welcome message styling */
    .welcome-message {
        font-size: 1.3rem; 
        color: #4CAF50; 
        margin-bottom: 20px;
        text-align: center;
        font-weight: 500;
    }
    
    /* Style improvements for chat interface */
    .stChatMessage {
        background-color: #f0f2f6 !important;
        border-radius: 10px !important;
        margin-bottom: 0.5rem !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
        padding: 0.75rem !important;
    }
    
    /* User messages vs AI messages */
    .stChatMessage[data-testid="stChatMessage-USER"] {
        background-color: #E3F2FD !important;
    }
    
    .stChatMessage[data-testid="stChatMessage-ASSISTANT"] {
        background-color: #F1F8E9 !important;
    }
    
    /* Ensure text messages have proper color contrast */
    .stChatMessage p, .stChatMessage div, .stChatMessage span {
        color: #111 !important;
    }
    
    /* Improve chat input styling */
    .stChatInputContainer {
        padding-top: 1rem !important;
        border-top: 1px solid #e0e0e0 !important;
    }
    
    /* Mobile-specific styling */
    @media (max-width: 768px) {
        [data-testid="stSidebar"] {display: none !important;}
        [data-testid="collapsedControl"] {display: none !important;}
        
        .main .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            max-width: 100% !important;
        }
        .welcome-message {
            font-size: 1.1rem;
            margin-bottom: 10px;
        }
        .stChatMessage {
            padding: 0.5rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Simple header - show first for instant feedback
    st.title(APP_TITLE)
    
    # Add welcoming message
    st.markdown('<p class="welcome-message">Hi! there I am here to help you</p>', unsafe_allow_html=True)
    
    # Initialize GUI components immediately
    chat_area = st.container()
    prompt_area = st.empty()
    
    # Initialize model in background
    if "model_loading" not in st.session_state:
        st.session_state.model_loading = True
        
        # Create a thread to load model in background
        import threading
        
        def load_model_in_background():
            # Download NLTK resources if needed
            download_nltk_resources()
            
            # Initialize database
            init_database()
            
            # Train sentiment model
            train_sentiment_model()
            
            # Mark as loaded
            st.session_state.model_loading = False
            st.session_state.model_loaded = True
        
        # Start background loading with smaller sample size
        threading.Thread(target=load_model_in_background).start()
    
    # Check for API key only once
    if not hasattr(st.session_state, "api_checked"):
        if not GEMINI_API_KEY:
            st.warning("⚠️ Gemini API Key not found or not set. Using local fallback responses.")
        st.session_state.api_checked = True
    
    # Initialize sentiment visualization in sidebar on desktop only
    with st.sidebar:
        st.subheader("Sentiment Analysis")
        
        # Show model loading status only if still loading
        if st.session_state.get("model_loading", True) and not st.session_state.get("model_loaded", False):
            with st.spinner("Initializing sentiment analysis..."):
                pass  # Just show the spinner without text
        
        # Show sentiment chart if we have data
        if len(st.session_state.sentiment_history) > 0:
            # Add visual color indicators for sentiment ranges
            st.markdown("""
            <style>
            .positive-sentiment { color: green; font-weight: bold; }
            .neutral-sentiment { color: gray; font-weight: bold; }
            .negative-sentiment { color: red; font-weight: bold; }
            </style>
            
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span class="negative-sentiment">Negative</span>
                <span class="neutral-sentiment">Neutral</span>
                <span class="positive-sentiment">Positive</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Create well-formatted chart with clear labels
            recent_sentiments = st.session_state.sentiment_history[-10:] if len(st.session_state.sentiment_history) > 10 else st.session_state.sentiment_history
            chart_data = pd.DataFrame({"Sentiment": recent_sentiments})
            
            # Add custom chart with better formatting
            st.line_chart(
                chart_data,
                use_container_width=True,
                height=200
            )
            
            # Display metrics in columns for better layout
            col1, col2 = st.columns(2)
            
            # Calculate average sentiment
            recent_sentiment = st.session_state.sentiment_history[-5:] if len(st.session_state.sentiment_history) >= 5 else st.session_state.sentiment_history
            avg_sentiment = sum(recent_sentiment) / len(recent_sentiment)
            
            with col1:
                st.metric("Avg Sentiment", f"{avg_sentiment:.2f}", get_sentiment_label(avg_sentiment))
            
            with col2:
                st.metric("Messages", len(st.session_state.messages) // 2)
            
            # Show sentiment trend analysis if we have enough data
            if len(st.session_state.messages) >= 2:
                trend = analyze_sentiment_trend(st.session_state.messages[-6:] if len(st.session_state.messages) >= 6 else st.session_state.messages)
                
                # Use color-coded trend indicators
                trend_color = {
                    "improving": "green",
                    "declining": "red",
                    "fluctuating": "orange",
                    "stable": "gray"
                }.get(trend, "gray")
                
                st.markdown(f"<p style='color:{trend_color}'>Recent trend: <b>{trend.capitalize()}</b></p>", unsafe_allow_html=True)
        else:
            # Add blank space placeholder for better layout instead of info message
            st.write("")
    
    # Display chat messages with optimized rendering
    with chat_area:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Show sentiment for user messages
                if message["role"] == "user" and message.get("sentiment") is not None:
                    sentiment_score = message["sentiment"]
                    sentiment_label = get_sentiment_label(sentiment_score)
                    
                    # Use color based on sentiment
                    if sentiment_score > 0.3:
                        color = "green"
                    elif sentiment_score > 0:
                        color = "lightgreen"
                    elif sentiment_score == 0:
                        color = "gray"
                    elif sentiment_score > -0.3:
                        color = "orange"
                    else:
                        color = "red"
                        
                    st.markdown(f"<span style='color:{color};font-size:0.8em'>Sentiment: {sentiment_label}</span>", unsafe_allow_html=True)
    
    # Input for new message
    prompt = st.chat_input("Type your message here...")
    if prompt:
        # Add user message to chat history
        if st.session_state.get("model_loaded", False):
            # Use trained model
            sentiment_score = analyze_sentiment(prompt)
        else:
            # Fallback to TextBlob until model is loaded
            sentiment_score = TextBlob(prompt).sentiment.polarity
            
        user_message = {"role": "user", "content": prompt, "sentiment": sentiment_score}
        st.session_state.messages.append(user_message)
        st.session_state.sentiment_history.append(sentiment_score)
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
            sentiment_label = get_sentiment_label(sentiment_score)
            
            # Use color based on sentiment
            if sentiment_score > 0.3:
                color = "green"
            elif sentiment_score > 0:
                color = "lightgreen"
            elif sentiment_score == 0:
                color = "gray"
            elif sentiment_score > -0.3:
                color = "orange"
            else:
                color = "red"
                
            st.markdown(f"<span style='color:{color};font-size:0.8em'>Sentiment: {sentiment_label}</span>", unsafe_allow_html=True)
        
        # Prepare messages for API call - only send necessary data
        api_messages = [generate_therapeutic_system_prompt(sentiment_score)]
        for msg in st.session_state.messages[-5:]:  # Only use last 5 messages for context
            if msg.get("sentiment") is not None:
                # For API calls, remove sentiment data
                api_messages.append({"role": msg["role"], "content": msg["content"]})
            else:
                api_messages.append(msg)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = call_gemini_api(api_messages)
                st.write(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Save conversation to database if model is ready
        if st.session_state.get("model_loaded", False):
            save_conversation(prompt, response, sentiment_score)

if __name__ == "__main__":
    main() 
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
import dask.dataframe as dd
from joblib import Memory
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from cachetools import TTLCache, cached
import asyncio
import httpx
from concurrent.futures import ThreadPoolExecutor

# Initialize caching
cachedir = '.cache'
memory = Memory(cachedir, verbose=0)
sentiment_cache = TTLCache(maxsize=1000, ttl=3600)  # Cache for 1 hour

# Load environment variables
load_dotenv()

# Constants
APP_TITLE = "Therapy AI"
APP_DESCRIPTION = "An AI-powered therapy chatbot that provides emotional support and guidance."
DB_PATH = "therapy_data.db"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
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

# Initialize device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize sentiment model
tokenizer = None
model = None

def initialize_sentiment_model():
    """Initialize the transformer-based sentiment model"""
    global tokenizer, model
    try:
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as e:
        st.error(f"Error loading sentiment model: {str(e)}")
        return None

@memory.cache
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
            tokens = [word for word in tokens if len(word) > 2]
        
        # Join tokens back into text
        return ' '.join(tokens)
    except Exception as e:
        print(f"Text preprocessing error: {str(e)}. Using original text.")
        return str(text)

@memory.cache
def process_dataset(file_path, sample_limit=None, chunk_size='50MB'):
    """Process dataset using Dask for better memory efficiency"""
    try:
        # Read data using Dask
        df = dd.read_csv(file_path, 
                        blocksize=chunk_size,
                        sample=sample_limit if sample_limit else None,
                        encoding='utf-8')
        
        # Convert to pandas for final processing
        df = df.compute()
        
        # Basic preprocessing
        df['ProcessedText'] = df['Text'].apply(preprocess_text)
        
        return df
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        return None

@cached(cache=sentiment_cache)
def analyze_sentiment_transformer(text):
    """Analyze sentiment using transformer model with caching"""
    try:
        # Tokenize and prepare input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Convert prediction to score between -1 and 1
        score = (predictions[0][1].item() - 0.5) * 2
        return score
    except Exception as e:
        print(f"Transformer model error: {str(e)}. Falling back to TextBlob.")
        return TextBlob(text).sentiment.polarity

async def call_gemini_api_async(messages):
    """Asynchronous version of Gemini API call with adaptive response length"""
    try:
        conversation_text = format_conversation_for_gemini(messages)
        
        # Analyze message length and complexity
        user_message = messages[-1]["content"] if isinstance(messages, list) else messages["content"]
        message_length = len(user_message.split())
        
        # Adjust max tokens based on message characteristics
        if message_length < 10:  # Very short message
            max_tokens = 50  # Brief response
        elif message_length < 20:  # Short message
            max_tokens = 100  # Moderate response
        else:  # Longer or more complex message
            max_tokens = 300  # Full response if needed
        
        payload = {
            "contents": [{
                "parts": [{"text": conversation_text}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": max_tokens,
                "topK": 40,
                "topP": 0.95,
                "stopSequences": ["User:", "Assistant:"]  # Prevent generating additional turns
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                GEMINI_API_URL,
                headers=headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            response_json = response.json()
            if 'candidates' in response_json and response_json['candidates']:
                response_text = response_json['candidates'][0]['content']['parts'][0]['text']
                # Clean up response
                response_text = response_text.strip()
                # Remove any system-style prefixes
                response_text = re.sub(r'^(Assistant:|Therapy AI:)\s*', '', response_text)
                return response_text
            
        raise ValueError("No response content found in API response")
            
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        # Use concise fallback responses
        sentiment_score = analyze_sentiment_transformer(messages[-1]["content"])
        return get_therapeutic_response(sentiment_score)

def download_nltk_resources():
    """Download necessary NLTK resources if not already present"""
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
    except Exception as e:
        st.warning(f"Error downloading NLTK resources: {str(e)}")

def train_sentiment_model():
    """Train a sentiment analysis model on multiple datasets with optimized processing"""
    if st.session_state.sentiment_model is not None:
        return st.session_state.sentiment_model, st.session_state.vectorizer
    
    try:
        # Process datasets using Dask
        twitter_df = process_dataset(TWITTER_DATA_PATH, sample_limit=5000)
        reviews_df = process_dataset(REVIEWS_DATA_PATH, sample_limit=5000)
        
        # Combine datasets if available
        combined_df = pd.concat([df for df in [twitter_df, reviews_df] if df is not None], 
                              ignore_index=True)
        
        if combined_df is None or len(combined_df) == 0:
            return None, None
        
        # Use optimized TfidfVectorizer settings
        vectorizer = TfidfVectorizer(
            max_features=2000,
            min_df=5,
            max_df=0.8,
            sublinear_tf=True,
            ngram_range=(1, 2)  # Include bigrams for better context
        )
        
        # Convert to sparse matrix for memory efficiency
        X = vectorizer.fit_transform(combined_df['ProcessedText'])
        y = combined_df['SentimentScore']
        
        # Train model with optimized settings
        model = LogisticRegression(
            max_iter=300,
            solver='saga',
            n_jobs=-1,
            C=1.0,
            class_weight='balanced'
        )
        
        # Use smaller test size for faster training
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Store in session state
        st.session_state.sentiment_model = model
        st.session_state.vectorizer = vectorizer
        
        # Initialize transformer model in background
        with ThreadPoolExecutor() as executor:
            executor.submit(initialize_sentiment_model)
        
        return model, vectorizer
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
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

def analyze_sentiment(text):
    """Analyze sentiment with fallback options and caching"""
    # Check cache
    if text in sentiment_cache:
        return sentiment_cache[text]
    
    # Try transformer model first
    if model is not None and tokenizer is not None:
        try:
            score = analyze_sentiment_transformer(text)
            if len(text) < 100:  # Cache only short texts
                sentiment_cache[text] = score
            return score
        except Exception as e:
            print(f"Transformer model error: {str(e)}. Trying custom model.")
    
    # Try custom trained model
    try:
        model = st.session_state.sentiment_model
        vectorizer = st.session_state.vectorizer
        
        if model is not None and vectorizer is not None:
            processed_text = preprocess_text(text)
            text_vec = vectorizer.transform([processed_text])
            score = model.predict(text_vec)[0]
            
            if len(text) < 100:
                sentiment_cache[text] = score
            return score
    except Exception as e:
        print(f"Custom model error: {str(e)}. Using TextBlob.")
    
    # Fallback to TextBlob
    score = TextBlob(text).sentiment.polarity
    if len(text) < 100:
        sentiment_cache[text] = score
    return score

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

def analyze_sentiment_trend(conversation_history):
    """Analyze sentiment trends with optimized processing"""
    # Get recent user messages
    recent_messages = [
        msg for msg in conversation_history[-6:]
        if msg["role"] == "user"
    ]
    
    if len(recent_messages) < 2:
        return "stable"
    
    # Calculate sentiment scores in parallel
    with ThreadPoolExecutor() as executor:
        scores = list(executor.map(
            lambda msg: msg.get("sentiment", analyze_sentiment(msg["content"])),
            recent_messages
        ))
    
    # Analyze trend
    if scores[-1] > scores[0] + 0.1:
        return "improving"
    elif scores[-1] < scores[0] - 0.1:
        return "declining"
    elif max(scores) - min(scores) > 0.3:
        return "fluctuating"
    else:
        return "stable"

def generate_therapeutic_system_prompt(sentiment_score):
    """Generate system prompt for the AI to act as a therapist with adaptive response length"""
    sentiment_category = get_sentiment_category(sentiment_score)
    
    base_prompt = """You are Therapy AI, a compassionate and empathetic AI therapist.
Your responses should be concise and adaptive to the conversation:
- For simple questions or acknowledgments, use very short responses (1-2 sentences)
- For emotional support or validation, use brief but warm responses (2-3 sentences)
- For complex situations or when specific guidance is needed, provide more detailed responses
- Always start with the most important point
- Break longer responses into short, digestible paragraphs
- Use natural, conversational language
- Never be verbose when a simple response would suffice

Additional guidelines:
- Listen carefully and validate feelings concisely
- Ask focused questions when needed
- Suggest practical strategies briefly
- Maintain a warm tone while being direct
- Never diagnose medical conditions
- For crisis situations, immediately suggest emergency services
"""
    
    # Add sentiment-specific guidance
    if sentiment_category == "very_negative":
        base_prompt += """The user is expressing very negative emotions:
- Respond with focused empathy
- Acknowledge their struggle briefly
- Offer immediate, practical support
- Keep responses gentle but direct"""
    elif sentiment_category == "somewhat_negative":
        base_prompt += """The user is expressing somewhat negative emotions:
- Validate feelings briefly
- Suggest one small, actionable step
- Balance support with gentle encouragement"""
    elif sentiment_category == "neutral":
        base_prompt += """The user's emotions appear neutral:
- Ask one clear, focused question
- Keep responses light and open-ended
- Use short, engaging responses"""
    elif sentiment_category == "somewhat_positive":
        base_prompt += """The user is expressing somewhat positive emotions:
- Reinforce positive elements briefly
- Ask about specific positive aspects
- Keep the momentum with short, encouraging responses"""
    elif sentiment_category == "very_positive":
        base_prompt += """The user is expressing very positive emotions:
- Celebrate briefly but meaningfully
- Ask about their success factors
- Keep the positive energy with concise responses"""
    
    return {"role": "system", "content": base_prompt}

async def process_user_message(message, conversation_history):
    """Process user message asynchronously"""
    # Analyze sentiment
    sentiment_score = analyze_sentiment(message)
    
    # Get API response
    api_response = await call_gemini_api_async({
        "role": "user",
        "content": message,
        "sentiment": sentiment_score
    })
    
    return sentiment_score, api_response

def get_therapeutic_response(sentiment_score):
    """Get a concise therapeutic response based on sentiment as fallback"""
    sentiment_category = get_sentiment_category(sentiment_score)
    
    # Concise fallback responses
    fallback_responses = {
        "very_negative": [
            "I hear how difficult this is. What would help you feel safer right now?",
            "That sounds really tough. Let's focus on one small step forward.",
            "I'm here with you. What do you need most in this moment?"
        ],
        "somewhat_negative": [
            "Things seem challenging. What might help, even a little?",
            "I understand your frustration. Should we explore some options?",
            "That's not easy to deal with. What's helped before?"
        ],
        "neutral": [
            "What's on your mind?",
            "Tell me more about that.",
            "How are you feeling about this?"
        ],
        "somewhat_positive": [
            "That's good to hear. What's working well?",
            "Sounds like you're making progress. What's helping?",
            "I'm glad things are looking up. What's next?"
        ],
        "very_positive": [
            "That's wonderful! What's contributing to this success?",
            "I'm so glad you're feeling good. How can we maintain this?",
            "Excellent progress! What's your next goal?"
        ]
    }
    
    # Choose a random concise response from the appropriate category
    responses = fallback_responses.get(sentiment_category, ["I'm here to listen. Please tell me more."])
    return random.choice(responses)

def main():
    """Main application with optimized performance"""
    # Configure app
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="auto"
    )
    
    # Add custom CSS for performance
    st.markdown("""
        <style>
        /* Optimize rendering */
        .stApp {
            contain: content;
        }
        /* Reduce reflow */
        .main {
            max-width: 1200px;
            margin: 0 auto;
            contain: layout;
        }
        /* Hardware acceleration */
        .stButton {
            transform: translateZ(0);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "sentiment_model" not in st.session_state:
        st.session_state.sentiment_model = None
    if "vectorizer" not in st.session_state:
        st.session_state.vectorizer = None
    
    # Simple header
    st.title(APP_TITLE)
    st.markdown('<p class="welcome-message">Hi! there I am here to help you</p>', 
                unsafe_allow_html=True)
    
    # Initialize GUI components
    chat_area = st.container()
    prompt_area = st.empty()
    
    # Initialize model in background if needed
    if "model_loading" not in st.session_state:
        st.session_state.model_loading = True
        
        def load_model_in_background():
            # Download NLTK resources
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            # Initialize models
            train_sentiment_model()
            initialize_sentiment_model()
            
            st.session_state.model_loading = False
            st.session_state.model_loaded = True
        
        with ThreadPoolExecutor() as executor:
            executor.submit(load_model_in_background)
    
    # Check API key
    if not hasattr(st.session_state, "api_checked"):
        if not GEMINI_API_KEY:
            st.warning("⚠️ Gemini API Key not found. Using local fallback responses.")
        st.session_state.api_checked = True
    
    # Display conversation history
    with chat_area:
        for message in st.session_state.conversation_history:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                st.markdown(f'<div class="user-message">{content}</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">{content}</div>', 
                          unsafe_allow_html=True)
    
    # Get user input
    with prompt_area:
        user_input = st.text_input("You:", key="user_input")
        
        if user_input:
            # Process message asynchronously
            sentiment_score, bot_response = asyncio.run(
                process_user_message(user_input, st.session_state.conversation_history)
            )
            
            # Update conversation history
            st.session_state.conversation_history.extend([
                {"role": "user", "content": user_input, "sentiment": sentiment_score},
                {"role": "assistant", "content": bot_response}
            ])
            
            # Clear input
            st.session_state.user_input = ""
            
            # Force rerun for immediate update
            st.experimental_rerun()

if __name__ == "__main__":
    main() 
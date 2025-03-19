# Therapy AI

A therapeutic chatbot with sentiment analysis powered by Gemini AI. This application uses natural language processing to provide supportive responses based on the detected sentiment of user messages.

## Features

- Real-time sentiment analysis of user messages
- Sentiment trend visualization
- Persistent conversation history
- Optimized performance with background model loading
- Fallback to local responses when API is unavailable
- Combined sentiment model trained on multiple datasets

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- NLTK
- Google Gemini AI API
- SQLite

## Demo

![Therapy AI Demo](Screenshot-therapy-ai-00)

## Installation

1. Clone this repository:
```
git clone https://github.com/Nithin-Adithya/therapy-ai.git
cd therapy-ai
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Google Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

4. Download the full datasets:
   - The full datasets are not included in this repository due to size limitations
   - You'll need to download the following datasets and place them in the project root:
     - Twitter dataset: Save as `twitter_training.csv`
     - Reviews dataset: Save as `Reviews.csv`
   - Sample versions of these datasets are provided in the `sample_data` folder

5. Run the application:
```
streamlit run main.py
```

## Datasets

The sentiment analysis model uses two combined datasets:
- Twitter dataset for social media sentiment
- Reviews dataset for product review sentiment

Sample data is included in the `sample_data` directory for reference.

## Project Structure

- `main.py`: Main application file
- `sample_data/`: Contains sample datasets for reference
- `therapy_data.db`: SQLite database for storing conversations (created on first run)
- `.env`: Environment variables (API keys)

## License

MIT License

## Acknowledgements

- Google for the Gemini AI API
- NLTK for natural language processing tools
- Streamlit for the web interface 

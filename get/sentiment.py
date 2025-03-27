import requests
from datetime import datetime, timedelta, timezone
import tensorflow as tf
from transformers import BertTokenizer
from training.trainsentiment.sentiment import clean_text

# Load the trained BERT model and tokenizer
model = tf.keras.models.load_model("bert_sentiment_model")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def analyze_sentiment(text):
    """Analyze sentiment of a single text using the trained BERT model."""
    # Preprocess the text
    text = clean_text(text)
    
    # Tokenize and prepare input for the model
    encodings = tokenizer([text], truncation=True, padding=True, max_length=128, return_tensors="tf")
    
    # Predict sentiment
    logits = model.predict(dict(encodings)).logits
    predicted_label = tf.argmax(logits, axis=1).numpy()[0]
    
    # Map the predicted label to a sentiment score
    if predicted_label == 2:  # Positive
        return 1.0
    elif predicted_label == 1:  # Neutral
        return 0.5
    else:  # Negative
        return 0.0

def get_sentiment(key, symbol):
    """Fetch news articles and calculate the average sentiment."""
    url = 'https://yahoo-finance166.p.rapidapi.com/api/news/list-by-symbol'
    querystring = {"s": symbol, "region": "US", "snippetCount": "5"}
    headers = {
        "x-rapidapi-host": "yahoo-finance166.p.rapidapi.com",
        "x-rapidapi-key": key
    }
    
    # Get the current date and time
    current_time = datetime.now(timezone.utc)
    # Define the threshold for filtering articles
    time_threshold = current_time - timedelta(days=7)

    filtered_data = []
    response = requests.get(url, params=querystring, headers=headers)
    data = response.json()
    stream_data = data.get('data', {}).get('main', {}).get('stream', [])
    
    for article in stream_data:
        content = article.get('content', {})
        title = content.get('title', 'No title available')
        pub_date_str = content.get('pubDate', 'No publication date available')

        try:
            pub_date = datetime.strptime(pub_date_str, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
        except ValueError:
            # If pubDate is not in the expected format, skip this item
            continue

        if pub_date > time_threshold:
            filtered_data.append(title)

    if not filtered_data:
        return 0  # No articles found, return neutral sentiment

    # Analyze sentiment for each title
    sum_sentiment = 0
    for title in filtered_data:
        sum_sentiment += analyze_sentiment(title)

    # Calculate the average sentiment
    average_sentiment = sum_sentiment / len(filtered_data)
    if average_sentiment > 0.5:
        sentiment = 1  # Positive sentiment
    else:
        sentiment = 0  # Negative sentiment

    return sentiment
import requests
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

import numpy as np
from datetime import datetime, timedelta, timezone
from sentiment import analyze_sentiment

TF_ENABLE_ONEDNN_OPTS = 0
API_KEY = "68221ded8dmsh6c87339388bb8d9p16f55djsn5e0f84453dcb"

def get_chart():
    url = 'https://yahoo-finance166.p.rapidapi.com/api/stock/get-chart?region=US&range=1d&symbol=AAPL&interval=5m'
    querystring = {"region":"US", "range":"1d", "symbol":"AAPL", "interval":"5m"}
    headers = {
        "x-rapidapi-host": "yahoo-finance166.p.rapidapi.com",
        "x-rapidapi-key": "68221ded8dmsh6c87339388bb8d9p16f55djsn5e0f84453dcb"
    }

def get_sentiment(symbol, sentiment):
    url = 'https://yahoo-finance166.p.rapidapi.com/api/news/list-by-symbol?s=AAPL%2CGOOGL%2CTSLA&region=US&snippetCount=500'
    querystring = {"s":symbol, "region":"US", "snippetCount": "5"}
    headers = {
        "x-rapidapi-host": "yahoo-finance166.p.rapidapi.com",
        "x-rapidapi-key": API_KEY
    }
    # Get the current date and time
    current_time = datetime.now(timezone.utc)
    # Define the threshold 
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

    sum = 0
    for title in filtered_data:
        sum += analyze_sentiment(title)

    average = sum / len(filtered_data)
    if average > 0.5:
        sentiment = 1  
    else:
        sentiment = 0  

    return sentiment

def get_prediction(sentiment, data, prediction):
    pass


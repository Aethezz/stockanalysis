import requests
from datetime import datetime, timedelta, timezone
from training.trainsentiment.sentiment import analyze_sentiment

def get_sentiment(key, symbol, sentiment):
    url = 'https://yahoo-finance166.p.rapidapi.com/api/news/list-by-symbol?s=AAPL%2CGOOGL%2CTSLA&region=US&snippetCount=500'
    querystring = {"s":symbol, "region":"US", "snippetCount": "5"}
    headers = {
        "x-rapidapi-host": "yahoo-finance166.p.rapidapi.com",
        "x-rapidapi-key": key
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
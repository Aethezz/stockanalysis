import requests
import datetime
import matplotlib.pyplot as plt
import io

def get_chart(key, symbol):
    url = 'https://yahoo-finance166.p.rapidapi.com/api/stock/get-chart'
    querystring = {"region":"US","range":"1mo","symbol":symbol, "interval":"1d"}
    headers = {
        "x-rapidapi-host": "yahoo-finance166.p.rapidapi.com",
        "x-rapidapi-key": key,
    }

    response = requests.get(url, params=querystring, headers=headers)

    if response.status_code == 200:
        data = response.json()
        
        try:
            # Extract timestamp and closing prices
            timestamps = data['chart']['result'][0]['timestamp']
            closing_prices = data['chart']['result'][0]['indicators']['quote'][0]['close']
            
            # Convert timestamps to human-readable format
            times = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
            
            # Plotting the chart
            plt.figure(figsize=(14, 7))
            plt.plot(times, closing_prices, marker='o', linestyle='-', label=f'{symbol} Price')
            plt.title(f'{symbol} Stock Price Over the Last Month')
            plt.xlabel('Time')
            plt.ylabel('Price (USD)')
            plt.grid(True)
            plt.legend()

            img_stream = io.BytesIO()
            plt.savefig(img_stream, format='png')
            plt.show()

            return img_stream.seek(0)
            
        except (KeyError, IndexError, TypeError) as e:
            print("Error processing chart data:", e)
    else:
        print("Failed to fetch data:", response.status_code, response.text)

    

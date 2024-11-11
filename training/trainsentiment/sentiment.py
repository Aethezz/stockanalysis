import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

def analyze_sentiment():
    pass

# Read the file into a DataFrame
df1 = pd.read_csv("training/trainsentiment/Sentences_75Agree.txt", header=None, sep=".@", encoding='latin1', names=["text", "sentiment"])

# Function to encode sentiment values with specific float values
def encode_sentiments_values(df):
    # Custom encoding values
    sentiment_dict = {
        "negative": 0.0,
        "neutral": 0.5,
        "positive": 1.0
    }

    # Replace the sentiment column with the new float values
    df["label"] = df.sentiment.replace(sentiment_dict)

    return df, sentiment_dict

# Apply the encoding function to your dataframe
df1, sentiment_dict = encode_sentiments_values(df1)

# Print the first few rows
print(df1.head())
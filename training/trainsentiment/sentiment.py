import sys
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import string
import nltk
import os
from transformers import AdamWeightDecay

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Ensure UTF-8 encoding for output
sys.stdout.reconfigure(encoding='utf-8')

# Download NLTK stopwords
nltk.download("stopwords")
from nltk.corpus import stopwords

# --- Load Dataset ---
df = pd.read_csv(
    "training/trainsentiment/Sentences_AllAgree.txt",
    header=None,
    sep=".@",
    engine="python",
    encoding="latin1",
    names=["text", "sentiment"]
)

# --- Preprocessing ---
# Clean non-printable characters
df["text"] = df["text"].apply(lambda x: ''.join(c for c in str(x) if c.isprintable()))

# Remove punctuation, stopwords, and lowercase
stop_words = set(stopwords.words("english"))

def clean_text(text):
    """Function to clean text: remove punctuation, stopwords, and lowercase."""
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(word.lower() for word in text.split() if word not in stop_words)
    return text

df["text"] = df["text"].apply(clean_text)

# Encode sentiments with one-hot encoding
sentiment_dict = {"negative": 0, "neutral": 1, "positive": 2}
df["label"] = df["sentiment"].replace(sentiment_dict).astype(int)

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# --- Load BERT Tokenizer and Model ---
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# --- Tokenize Input Data ---
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128, return_tensors="tf")
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128, return_tensors="tf")

# --- Prepare the Dataset ---
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
)).batch(16)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_test
)).batch(16)

# --- Compile the Model ---
optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
model.compile(optimizer=optimizer, loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

# --- Train the Model ---
history = model.fit(
    train_dataset,
    epochs=3,
    validation_data=test_dataset
)

# Save the trained model
model.save("bert_sentiment_model")

# --- Evaluate the Model ---
loss, accuracy = model.evaluate(test_dataset)
print(f"\n✅ Test Loss: {loss:.4f}")
print(f"✅ Test Accuracy: {accuracy:.4f}")

# --- Test on New Dataset ---
test_df = pd.read_csv(
    "training/trainsentiment/Sentences_75Agree.txt",
    header=None,
    sep=".@",
    engine="python",
    encoding="latin1",
    names=["text", "sentiment"]
)

# Preprocess the new dataset
test_df["text"] = test_df["text"].apply(lambda x: ''.join(c for c in str(x) if c.isprintable()))
test_df["text"] = test_df["text"].apply(clean_text)

# Tokenize and pad the new dataset
new_test_encodings = tokenizer(list(test_df["text"]), truncation=True, padding=True, max_length=128, return_tensors="tf")

# Make predictions
predictions = model.predict(dict(new_test_encodings)).logits
predicted_labels = tf.argmax(predictions, axis=1).numpy()

# Map predictions to sentiment labels
sentiment_labels = ["Negative", "Neutral", "Positive"]
test_df["predicted_sentiment"] = [sentiment_labels[label] for label in predicted_labels]
test_df["actual_sentiment"] = test_df["sentiment"].map({0: "Negative", 1: "Neutral", 2: "Positive"})

# Display results
result_df = test_df[["text", "actual_sentiment", "predicted_sentiment"]]
print("\n📊 **Predictions on New Dataset:**")
print(result_df.head())

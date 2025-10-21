# app.py
import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re
import numpy as np

# -----------------------------------------

def clean_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = tweet.lower()
    return tweet

# -----------------------------------------

tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))  # احفظ Tokenizer بعد التدريب
model = load_model('best_model1.h5')

# -----------------------------------------
st.title("Twitter Sentiment Analysis")
st.write("Predict the sentiment of a tweet (Positive, Neutral, Negative)")

tweet_input = st.text_area("Enter your tweet here:")

if st.button("Predict"):
    if tweet_input.strip() == "":
        st.warning("Please enter a tweet to predict.")
    else:
        seq = tokenizer.texts_to_sequences([clean_tweet(tweet_input)])
        padded = pad_sequences(seq, maxlen=50)
        pred = model.predict(padded)
        class_idx = np.argmax(pred, axis=1)[0]
        label_map_reverse = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        sentiment = label_map_reverse[class_idx]
        st.success(f"Predicted Sentiment: {sentiment}")

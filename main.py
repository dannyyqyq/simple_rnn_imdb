import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Step 1 : Load model and dataset
# Load IMDB dataset word index
word_index = imdb.get_word_index()

# Reverse word index
reverse_word_index = {value: key for key, value in word_index.items()}
# Load model
model = load_model('simple_rnn_model.keras')

# Step 2 : Load helper functinos
# Helper function to decode reviews
def decode_review(review):
    """
    This function takes an integer-encoded review and converts it back to readable text.
    """
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in review])

# preprocess user input
def preprocess_text(text):
    """
    This function takes raw text input (e.g., "The movie was great") and preprocesses it into the same format as 
    X_train - a padded integer sequence—for use with your RNN model.
    """
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Step 3 : Make predictions

# streamlit app
st.title("IMDB movie review sentiment analysis")
st.write("Enter a movie review to classify it as positive or negative.")

#User input
user_input = st.text_area("Enter your movie review here:")

if st.button("Classify"):
    preprocess_input = preprocess_text(user_input)
    prediction = model.predict(preprocess_input)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

    # Display result
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction: {prediction[0][0]}")
else:
    st.write("Please enter a movie review")
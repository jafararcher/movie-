import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🎬 Movie Rating Predictor")

# Define genres (use the same ones from your training script)
genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Thriller']  # example list

# Collect input
genre_input = [st.checkbox(genre) for genre in genres]
avg_rating = st.slider("Average Movie Rating", 0.0, 5.0, 3.0)
rating_count = st.number_input("Rating Count", min_value=0, value=100)

# Feature vector
features = np.array(genre_input + [avg_rating, rating_count]).reshape(1, -1)
features_scaled = scaler.transform(features)

if st.button("Predict"):
    prediction = model.predict(features_scaled)[0]
    st.success(f"Predicted Rating: ⭐ {round(prediction, 2)}")

import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🎬 Movie Rating Predictor")

# Genres used in training
genres = ['Comedy', 'Drama', 'Thriller', 'Romance', 'Action']
genre_input = [1 if st.checkbox(g) else 0 for g in genres]

# Additional features
avg_rating = st.slider("Average Movie Rating", 0.0, 5.0, 3.0, 0.1)
rating_count = st.number_input("Number of Ratings", min_value=0, step=1, value=100)

# Final input vector (must match training order)
features = np.array(genre_input + [avg_rating, rating_count]).reshape(1, -1)

# Display shape to debug
st.write(f"Feature vector shape: {features.shape}")  # should be (1, 7)

# Make prediction
features_scaled = scaler.transform(features)
prediction = model.predict(features_scaled)[0]

# Show result
st.success(f"📊 Predicted Movie Rating: ⭐ {round(prediction, 2)}")

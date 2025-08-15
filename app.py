# app.py
import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# Load the model and vectorizer
# The `sentiment_model.pkl` and `vectorizer.pkl` files are created by train_model.py
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page configuration for a clean layout
st.set_page_config(page_title="Emotion Classifier", page_icon="😊", layout="centered")

# Custom CSS for styling the app
st.markdown(
    """
    <style>
        .main {
            background-color: #f5f7fa;
        }
        .title {
            font-size: 2.2rem;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
        }
        .result {
            font-size: 1.5rem;
            font-weight: bold;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        .positive {
            background-color: #d4edda;
            color: #155724;
        }
        .negative {
            background-color: #f8d7da;
            color: #721c24;
        }
        .neutral {
            background-color: #fff3cd;
            color: #856404;
        }
        .history-title {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 2rem;
        }
        .st-emotion-cache-1g821v { /* This is a Streamlit class for DataFrame headers, which can change */
            color: #2c3e50;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Application title and description
st.markdown("<div class='title'>😊 Emotion Classifier App</div>", unsafe_allow_html=True)
st.write("This app predicts the **emotion** of a given text using a trained ML model.")

# Initialize session state for history if it doesn't exist
if 'history' not in st.session_state:
    st.session_state.history = []

# Text area for user input
user_input = st.text_area("Enter your text here:", placeholder="Type something...")

# Button to trigger the prediction
if st.button("Predict Emotion"):
    # Check if the input is not empty
    if user_input.strip() != "":
        # Transform the user's text into a numerical vector using the trained vectorizer
        input_vector = vectorizer.transform([user_input])
        
        # Predict the emotion using the loaded model
        prediction = model.predict(input_vector)[0]

        # Display the prediction with appropriate styling based on the result
        if prediction.lower() == "positive":
            st.markdown(f"<div class='result positive'>Prediction: {prediction}</div>", unsafe_allow_html=True)
        elif prediction.lower() == "negative":
            st.markdown(f"<div class='result negative'>Prediction: {prediction}</div>", unsafe_allow_html=True)
        elif prediction.lower() == "neutral":
            st.markdown(f"<div class='result neutral'>Prediction: {prediction}</div>", unsafe_allow_html=True)
        else:
            # Fallback for any other prediction
            st.markdown(f"<div class='result neutral'>Prediction: {prediction}</div>", unsafe_allow_html=True)

        # Add the prediction to the session state history
        st.session_state.history.append({
            "Text": user_input,
            "Prediction": prediction,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    else:
        # Show a warning if no text is entered
        st.warning("Please enter some text to predict.")

# Display history if it exists
if st.session_state.history:
    st.markdown("<div class='history-title'>Prediction History</div>", unsafe_allow_html=True)
    
    # Create a DataFrame from the history list
    history_df = pd.DataFrame(st.session_state.history)

    # Display the DataFrame, sorted by timestamp in descending order
    st.dataframe(history_df.sort_values(by="Timestamp", ascending=False), hide_index=True)

    # Add a button to clear the history
    if st.button("Clear History"):
        st.session_state.history = []
        st.experimental_rerun()

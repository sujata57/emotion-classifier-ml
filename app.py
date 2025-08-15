import streamlit as st
import joblib
import pandas as pd
import sqlite3

# --- File Paths (Update these as needed) ---
MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
DB_PATH = "sentiment_history.db"

# --- Function to Initialize the Database ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            prediction TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# --- Function to Save Prediction to DB ---
def save_prediction(text, prediction):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO predictions (text, prediction) VALUES (?, ?)", (text, prediction))
    conn.commit()
    conn.close()

# --- Function to Load History from DB ---
def load_history():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    conn.close()
    return df

# --- Function to Clear History from DB ---
def clear_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    st.info("History cleared!")
    st.rerun() # Use st.rerun() to refresh the app state

# --- Main App Logic ---
if __name__ == "__main__":
    # Load the model and vectorizer
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    except FileNotFoundError:
        st.error("Model files not found. Please run `train_model.py` first.")
        st.stop()

    # Initialize the database
    init_db()

    # Page config
    st.set_page_config(page_title="Emotion Classifier", page_icon="😊", layout="centered")

    # Custom CSS for styling
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
                margin-top: 20px;
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
            .stTextArea {
                border-radius: 8px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title
    st.markdown("<div class='title'>😊 Emotion Classifier App</div>", unsafe_allow_html=True)
    st.write("This app predicts the **emotion** of a given text using a trained ML model.")

    # Text input
    user_input = st.text_area("Enter your text here:", placeholder="Type something...")

    # Prediction
    if st.button("Predict Emotion"):
        if user_input.strip() != "":
            # Transform input
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)[0]

            # Display result and save to history
            if prediction.lower() == "positive":
                st.markdown(f"<div class='result positive'>Prediction: {prediction}</div>", unsafe_allow_html=True)
            elif prediction.lower() == "negative":
                st.markdown(f"<div class='result negative'>Prediction: {prediction}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='result neutral'>Prediction: {prediction}</div>", unsafe_allow_html=True)
            
            save_prediction(user_input, prediction)
            st.session_state.show_history = True
            
        else:
            st.warning("Please enter some text to predict.")

    # History Section
    st.markdown("---")
    st.markdown("### Prediction History")
    
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Show History"):
            st.session_state.show_history = True
    
    with col2:
        if st.button("Clear History"):
            clear_history()

    if st.session_state.get("show_history", False):
        history_df = load_history()
        if not history_df.empty:
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("No prediction history yet.")

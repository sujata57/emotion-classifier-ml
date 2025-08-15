# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Load the new, larger dataset from the specified path.
# The raw string prefix 'r' is used to handle backslashes correctly in the file path.
# We've added the 'encoding' parameter to handle the UnicodeDecodeError.
data_path = r"C:\Users\91738\OneDrive\SUJATA\sentiment_app\sentiment.csv"
df = pd.read_csv(data_path, encoding='latin-1')

print("Dataset loaded successfully!")
print("First 5 rows:\n", df.head())

# 2. Define the text and label columns for the new dataset
# The 'text' column contains the full tweet text.
# The 'sentiment' column contains the labels ('positive', 'negative', 'neutral').
text_column = "text"
label_column = "sentiment"

# We will drop any rows where the 'text' or 'sentiment' is missing to avoid errors.
df.dropna(subset=[text_column, label_column], inplace=True)

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df[text_column], df[label_column], test_size=0.2, random_state=42
)

# 4. Vectorize the text data using TF-IDF
# This converts text into numerical features that the model can understand.
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train a Logistic Regression model
# Logistic Regression is a good baseline model for text classification.
model = LogisticRegression(max_iter=1000) # Increased max_iter for better convergence on larger data
model.fit(X_train_vec, y_train)

# 6. Evaluate the model (optional but recommended for a better project)
from sklearn.metrics import accuracy_score, classification_report
predictions = model.predict(X_test_vec)
print("\nModel Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))


# 7. Save the trained model and vectorizer to disk
# These files will be loaded by the Streamlit app to make predictions.
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nNew model and vectorizer saved successfully!")

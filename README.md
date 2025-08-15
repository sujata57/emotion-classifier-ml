Emotion Classifier Web App using Machine Learning


Project Description

This project is an end-to-end machine learning application that classifies the emotion or sentiment of a given text input. Built as a web application using Streamlit, this tool provides a simple and intuitive interface for real-time predictions. The core of the app is a text classification model trained on a large dataset of social media text, demonstrating a complete and reproducible machine learning workflow.

Tech Stack

Python: The primary programming language used for the entire project.

Streamlit: A Python library used to create the interactive web application interface.

scikit-learn: The machine learning library for model training and text vectorization.

pandas: Used for data manipulation and analysis of the dataset.

joblib: A tool for saving and loading the trained machine learning model and vectorizer.



Setup and Installation

To get this project up and running on your local machine, follow these steps:

1. Clone the Repository
   
First, clone the project from GitHub to your local machine using the command line.

git clone https://github.com/your-username/your-repository-name.git cd your-repository-name

2. Install Dependencies
   
Make sure you have Python installed. Then, install the required libraries using pip.

pip install -r requirements.txt

3. Train the Model
   
The app relies on a trained model. Run the train_model.py script to generate the necessary .pkl files (sentiment_model.pkl and vectorizer.pkl).

python train_model.py

4. Run the Streamlit App
   
Once the model is trained, you can launch the web application.

python -m streamlit run app.py

The app will open automatically in your web browser.


Example Usage

Simply enter a piece of text into the text box and click the "Predict Emotion" button. The application will instantly display the predicted sentiment.


Example Input:

"I am so happy with this result! It's fantastic."

Example Output:

Prediction: Positive


File Structure

app.py: The main Streamlit application file.

train_model.py: The script used to train and save the machine learning model.

sentiment.csv: The dataset used for training the model.

requirements.txt: A list of all necessary Python libraries.

sentiment_model.pkl: The saved machine learning model.

vectorizer.pkl: The saved TF-IDF vectorizer.


Future Improvements

Expand the model to predict a wider range of emotions (e.g., anger, fear, joy).

Add a feature to visualize sentiment over time for a series of inputs.

Integrate a user feedback loop to collect new data and improve model accuracy.

Deploy the app to a permanent cloud hosting service like Streamlit Community Cloud.


Author
sujata57

# Twitter Sentiment Analysis (RNN/LSTM)

A deep learning project to classify tweets into **Positive, Neutral, or Negative** sentiments using RNN/LSTM.

---

## 📌 Overview
This project builds a **Twitter Sentiment Analysis** system using **RNN (SimpleRNN)** to classify tweets into **Positive, Neutral, or Negative** sentiments.  
The model is trained on a Twitter dataset and can predict the sentiment of new tweets in real-time using a Streamlit app.

---

## 🛠 Features
- Text preprocessing: clean, lowercase, remove stopwords, tokenize
- Word embeddings via **Keras Tokenizer**
- SimpleRNN model with Dropout for regularization
- EarlyStopping & ModelCheckpoint during training
- Evaluation with accuracy, loss, and classification report
- Streamlit web app for live prediction

---

## 🗂 Project Structure

Twitter-Sentiment-RNN/
│
├── app.py # Streamlit application
├── RNN.ipynb # Training notebook
├── best_model1.h5 # Trained RNN model
├── tokenizer.pkl # Saved Keras Tokenizer
├── requirements.txt # Required packages
├── images/ # Screenshots for README
│ ├── negative.png
│ ├── neutral.png
│ ├── positive.png
└── README.md # Project documentation

yaml
Copy code

---

## ⚙ Installation

### 1. Clone the repository
```bash
git clone https://github.com/username/Twitter-Sentiment-RNN.git
cd Twitter-Sentiment-RNN
```
### 2. Install dependencies
 
pip install -r requirements.txt
## 🚀 How to Run the Streamlit App
 
streamlit run app.py
Open the link in your browser and enter a tweet to predict its sentiment.

## 🧠 Usage Example
In Streamlit app

Enter Tweet: "I love this game!"
Predicted Sentiment: Positive
📊 Model Performance
Validation Accuracy: ~93%

Loss: ~0.21

Classification Report shows good performance for all classes: Negative, Neutral, Positive

## 🔗 Live Demo
Streamlit Live Demo  https://twittersentimentanalysisrnn-3yt4jdbfzpwj7itdryjrmc.streamlit.app/

 
## 📸 Screenshots

### 🟥 Negative Tweet Prediction
![Negative](image/Screenshot%202025-10-21%20121348.png)

### 🟨 Neutral Tweet Prediction
![Neutral](image/Screenshot%202025-10-21%20122002.png)

### 🟩 Positive Tweet Prediction
![Positive](image/Screenshot%202025-10-21%20122158.png)


📦 Notes
Dataset files are not included due to size limitations.

Ensure best_model1.h5 and tokenizer.pkl are in the same folder as app.py for Streamlit.

You can retrain the model using RNN.ipynb if desired.

✨ Author
Mahmoud Ahmed Mahmoud Attia

# IMDB Movie Review Sentiment Analysis

This project is a **sentiment analysis web application** that classifies movie reviews as **positive or negative** using a **Recurrent Neural Network (RNN)** trained on the IMDB dataset. The model is deployed using **Streamlit** for an interactive user experience.

## 🚀 Features

- Loads a **pre-trained RNN model** (`simple_rnn_model.h5`)
- Preprocesses user input by **tokenizing and padding**
- Classifies reviews as **positive or negative**
- Provides a **probability score** for sentiment prediction
- Simple **web interface** using **Streamlit**

## 📂 Project Structure

```bash
├── main.py                 # Streamlit web app for sentiment analysis
├── simple_rnn_model.h5     # Pre-trained RNN model
├── README.md               # Project documentation
```

## 🔧 Installation & Setup

### 1️⃣ Install Dependencies

Make sure you have Python installed, then install the required libraries:

```sh
pip install -r requirements.txt
```

### 2️⃣ Run the Streamlit App

```sh
streamlit run main.py
```

## 🛠 How It Works

1. **Loads the IMDB dataset word index** for text processing.
2. **Preprocesses user input** by converting words into integers and padding sequences.
3. **Uses a trained RNN model** to classify the sentiment of the review.
4. **Displays the sentiment (Positive/Negative) and confidence score** on the web UI.

## 📌 Example Usage

```plaintext
Input: "The movie was fantastic! I loved it."
Output: Sentiment: Positive with a confidence score.
```

## 🤖 Model Training (Optional)

If you want to train your own model, you can use TensorFlow and the IMDB dataset to create and save an RNN model.

## 📜 License

This project is for educational purposes. Feel free to modify and use it as needed.

---

**Happy Coding! 🎬📊**


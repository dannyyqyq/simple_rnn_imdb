# IMDB Movie Review Sentiment Analysis

This project is a **sentiment analysis web application** that classifies movie reviews as **positive** or **negative** using a **Recurrent Neural Network (RNN)** trained on the IMDB dataset. It’s deployed as an interactive app using **Streamlit**, blending machine learning with a user-friendly interface.

## 🌐 Live Demo

Try it out: [IMDB Movie Review Classifier](https://imdb-movie-review-classification.streamlit.app/)  
Enter a movie review and get an instant sentiment prediction with a confidence score!

## 🚀 Features

- Loads a **pre-trained RNN model** (`simple_rnn_model.h5`).
- Preprocesses input by **tokenizing and padding** text to match the training format.
- Classifies reviews as **positive** or **negative** with a **probability score**.
- Offers a clean, interactive **web interface** via Streamlit.

## 📂 Project Structure

## 🔧 Installation & Setup

### Prerequisites
- Python 3.12
- Git

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/dannyyqyq/simple_rnn_imdb
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the App Locally
```sh
streamlit run main.py
```

## 🛠 How It Works

1. Loads the **IMDB dataset word index** for text preprocessing.
2. Converts user input into **integer sequences** and pads them to 500 tokens (pre-padding).
3. Uses a **trained SimpleRNN model** to predict sentiment.
4. Displays the result (**Positive** or **Negative**) with a confidence score on the Streamlit UI.

## 📌 Example Usage

**Input**: "The movie was fantastic! I loved it."  
**Output**: Sentiment: Positive, Confidence: ~0.75 (example score)

## 🤖 Model Details

- **Dataset**: IMDB movie reviews (25,000 training samples, 10,000-word vocabulary).
- **Architecture**: 
  - `Embedding` layer: 10,000 vocab size, 128D output, 500 timesteps.
  - `SimpleRNN`: 128 units, ReLU activation.
  - `Dense`: 1 unit, sigmoid activation.
- **Training**: 10 epochs with early stopping, ~90% validation accuracy.
- **Padding**: Pre-padding to 500 tokens.

To retrain the model, refer to the training notebook (e.g., `train_model.ipynb`) if included in the repo.

## 📜 License

This project is open-source under the [MIT License](LICENSE). Feel free to adapt and use it!

## 📢 Notes
- The model achieves moderate accuracy (~65-71%), so predictions may vary with nuanced or creative reviews.
- Built as part of my machine learning journey!

**Happy Coding! 🎬📊**  

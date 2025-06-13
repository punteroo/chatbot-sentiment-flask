from flask import Flask, request, render_template
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Load model and vectorizer
with open('model/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load Spanish stopwords
stop_words = set(stopwords.words('spanish'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    if not review:
        return render_template('index.html', prediction='Por favor, ingrese una reseña.', sentiment_class='neutral')

    # Preprocess and vectorize input
    cleaned_review = preprocess_text(review)
    review_tfidf = vectorizer.transform([cleaned_review])

    # Predict sentiment
    prediction = model.predict(review_tfidf)[0]

    # Map internal labels to display labels
    display_label = {
        'great': 'Excelente',
        'normal': 'Neutral',
        'bad': 'Mala'
    }.get(prediction, 'Regular')

    # Assign CSS class for color coding
    sentiment_class = {
        'great': 'positive',
        'normal': 'neutral',
        'bad': 'negative'
    }.get(prediction, 'neutral')

    return render_template('index.html', prediction=f"Sentimiento: {display_label}", 
                         sentiment_class=sentiment_class, original_review=review)

if __name__ == '__main__':
    app.run(debug=True)
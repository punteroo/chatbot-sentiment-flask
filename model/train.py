import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load Spanish stopwords
stop_words = set(stopwords.words('spanish'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Load dataset (replace with your dataset path)
data = pd.read_csv('model/dataset/dataset_sentiment_analisys.csv', delimiter=";")

# Map Spanish sentiment labels to English labels
def map_sentiment_to_label(sentiment):
    mapping = {'positivo': 'great', 'neutral': 'normal', 'negativo': 'bad'}
    return mapping.get(sentiment.lower(), 'normal')  # Default to 'normal' if unknown

data['label'] = data['sentimiento'].apply(map_sentiment_to_label)

# Preprocess reviews
data['cleaned_review'] = data['review'].apply(preprocess_text)

# Split data
X = data['cleaned_review']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate model
y_pred = model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
with open('model/sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
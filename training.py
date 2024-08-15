import pandas as pd
import numpy as np
import string
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pickle

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv("REVIEW.csv")

# Data preprocessing
df["Patient's Review"] = df["Patient's Review"].str.lower()

# List of stopwords
stopwordlist = nltk.corpus.stopwords.words('english')

# Function to remove stopwords
def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in set(stopwordlist)])

df["Patient's Review"] = df["Patient's Review"].apply(lambda text: cleaning_stopwords(text))

# Function to remove punctuations
def cleaning_punctuations(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

df["Patient's Review"] = df["Patient's Review"].apply(lambda x: cleaning_punctuations(x))

# Function to remove numbers
def cleaning_numbers(data):
    return re.sub('[0-9]+', "", data)

df["Patient's Review"] = df["Patient's Review"].apply(lambda x: cleaning_numbers(x))

# Tokenization
tokenizer = nltk.RegexpTokenizer(r'\w+')
df["Patient's Review"] = df["Patient's Review"].apply(tokenizer.tokenize)

# Lemmatization
lm = nltk.WordNetLemmatizer()
def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return text

df["Patient's Review"] = df["Patient's Review"].apply(lambda x: lemmatizer_on_text(x))

# Convert the tokenized words back into strings
x = df["Patient's Review"].apply(lambda x: " ".join(x)).values
y = df["Class"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# TF-IDF Vectorizer
vectoriser = TfidfVectorizer(ngram_range=(1, 2), max_features=500000)
vectoriser.fit(X_train)

X_train = vectoriser.transform(X_train)
X_test = vectoriser.transform(X_test)

# Training the SVC model
SVCmodel = LinearSVC()
SVCmodel.fit(X_train, y_train)

# Save the model and vectorizer
with open('model/svc_model.pkl', 'wb') as f:
    pickle.dump(SVCmodel, f)

with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectoriser, f)

print("Model training complete and saved as svc_model.pkl and vectorizer.pkl")

from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model and vectorizer
with open('model/svc_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/vectorizer.pkl', 'rb') as f:
    vectoriser = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        
        # Transform the review text to a vector
        review_vector = vectoriser.transform([review])
        
        # Make a prediction
        prediction = model.predict(review_vector)
        
        # Determine the sentiment based on the prediction
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'

        # Render the result
        return render_template('index.html', 
                               prediction=sentiment, 
                               review=review)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)


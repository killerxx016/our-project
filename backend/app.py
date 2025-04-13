from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(vectors)[0][1]
    return float(similarity)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check_plagiarism', methods=['POST'])
def check_plagiarism():
    data = request.get_json()
    text1 = data.get('text1', '')
    text2 = data.get('text2', '')
    
    similarity = calculate_similarity(text1, text2)
    percentage = round(similarity * 100, 2)
    
    return jsonify({
        'similarity': percentage,
        'message': f'Benzerlik oranı: {percentage}%'
    })

if __name__ == '__main__':
    app.run(debug=True)
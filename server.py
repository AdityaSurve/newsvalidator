from flask import Flask, request, jsonify
from DataIngestion import NewsApi
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
from nltk.corpus import wordnet
from string import punctuation
import re
import numpy as np
import pandas as pd

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def preprocess_articles(articles):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    preprocessed_articles = []
    for article in articles:
        article = re.sub(r'\s+', ' ', article)
        article = re.sub(r'[^\w\s]', '', article)
        article = article.lower()

        tokens = word_tokenize(article)
        tokens = [token for token in tokens if token not in stop_words]

        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        preprocessed_articles.append(' '.join(lemmatized_tokens))

    return preprocessed_articles


def summarize_cluster(cluster_articles):
    articles_by_title = {}
    for article in cluster_articles:
        title = article['title']
        if title not in articles_by_title:
            articles_by_title[title] = []
        articles_by_title[title].append(article)

    analysis = []
    for title, articles in articles_by_title.items():
        descriptions = [article['description'] for article in articles]
        description_counts = {description: descriptions.count(
            description) for description in descriptions}

        analysis.append({
            'title': title,
            'numberOfArticles': len(articles),
            'oldestArticle': min(articles, key=lambda x: x['publishedAt']),
            'latestArticle': max(articles, key=lambda x: x['publishedAt']),
            'summary': description_counts
        })

    return analysis

app = Flask(__name__)
CORS(app)
newsapi = NewsApi()


@app.route('/newsapi', methods=['GET'])
def get_newsapi():
    category = request.args.get('category')
    source = request.args.get('source')
    query = request.args.get('query')
    language = request.args.get('language')
    country = request.args.get('country')
    date = request.args.get('date')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    news = newsapi.get_news(category, source, query,
                            language, country, date, start_date, end_date)

    if len(news) == 0:
        return jsonify({'error': 'No articles found'})
    if len(news) < 5:
        return jsonify(news)
    articles = [article['content'] for article in news['articles']]
    articles = preprocess_articles(articles)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(articles)

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X)
    clusters = kmeans.predict(X)

    analysis = []
    for i in range(5):
        cluster_articles = [article for j, article in enumerate(
            news['articles']) if clusters[j] == i]
        analysis.append(summarize_cluster(cluster_articles))

    news['analysis'] = analysis

    return jsonify(news)

if __name__ == '__main__':
    app.run(debug=True)

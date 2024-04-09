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
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import re
from sklearn.metrics.pairwise import linear_kernel
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine

BOOST_VALUE = 0.2
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

def calculate_similarity(vector1, vector2):
    return linear_kernel(vector1, vector2).flatten()[0]

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
        if article is None or article.strip() == '':
            continue
        article = re.sub(r'\s+', ' ', article)
        article = re.sub(r'[^\w\s]', '', article)
        article = article.lower()

        tokens = word_tokenize(article)
        tokens = [token for token in tokens if token not in stop_words]

        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        preprocessed_articles.append(' '.join(lemmatized_tokens))

    return preprocessed_articles


def group_similar_articles(articles):
    preprocessed_articles = preprocess_articles(
        [article['content'] for article in articles])

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(preprocessed_articles)

    similarity_matrix = cosine_similarity(X)

    clustering_model = AgglomerativeClustering(
        n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=0.5)
    labels = clustering_model.fit_predict(1 - similarity_matrix)

    grouped_articles = {}
    for i, label in enumerate(labels):
        if label not in grouped_articles:
            grouped_articles[label] = []
        grouped_articles[label].append(articles[i])

    return grouped_articles


def summarize_cluster(articles):
    if len(articles) == 0:
        return {
            'numberOfArticles': 0,
            'firstArticle': None,
            'lastArticle': None,
            'sources': [],
            'descriptionGroups': ""
        }
    if len(articles) < 2:
        return {
            'numberOfArticles': len(articles),
            'firstArticle': articles[0],
            'lastArticle': articles[0],
            'sources': [articles[0]['source']['name']],
            'description': articles[0]['description']
        }
    descriptions = [article['description'] for article in articles]
    preprocessed_descriptions = preprocess_articles(descriptions)
    if len(preprocessed_descriptions) == 0:
        return {
            'numberOfArticles': len(articles),
            'firstArticle': articles[0],
            'lastArticle': articles[-1],
            'sources': [article['source']['name'] for article in articles],
            'description': descriptions[0]
        }
    sources = [article['source']['name'] for article in articles]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(preprocessed_descriptions)
    similarity_matrix = cosine_similarity(X)
    clustering_model = AgglomerativeClustering(
        n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=0.5)
    labels = clustering_model.fit_predict(1 - similarity_matrix)

    description = ""
    for i, label in enumerate(labels):
        label = int(label)
        description = descriptions[i]

    return {
        'numberOfArticles': len(articles),
        'firstArticle': articles[0],
        'lastArticle': articles[-1],
        'sources': sources,
        'description': description,
    }


# def get_embedding(text):
#     inputs = tokenizer(text, return_tensors='pt')
#     outputs = model(**inputs)
#     return outputs.last_hidden_state.squeeze().detach().numpy()

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

# def get_embedding(text):
#     inputs = tokenizer(text, return_tensors='pt')
#     outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).detach().numpy()

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

    if len(news['articles']) == 0:
        return jsonify({'error': 'No articles found'})
    if len(news['articles']) < 5:
        return jsonify(news)

    query_embedding = get_embedding(query)

    for i, article in enumerate(news['articles']):
        description = article['description']
        title = article['title']
        if description is not None and description.strip() != '':
            description_embedding = get_embedding(description)
            similarity_desc = 1 - \
                cosine(query_embedding, description_embedding)
        if title is not None and title.strip() != '':
            title_embedding = get_embedding(title)
            similarity_title = 1 - cosine(query_embedding, title_embedding)
        avg_similarity = min(
            ((similarity_desc + similarity_title) / 2) + BOOST_VALUE, 1)
        news['articles'][i]['relevanceScore'] = avg_similarity

    filtered_articles = [i for i in news['articles']
                         if 'relevanceScore' in i and i['relevanceScore'] > 0.5]
    filtered_articles = sorted(
        filtered_articles, key=lambda x: x['relevanceScore'], reverse=True)
    news['articles'] = filtered_articles
    analysis = []
    grouped_articles = group_similar_articles(news['articles'])
    for articles in grouped_articles.values():
        analysis.append(summarize_cluster(articles))
    news['analysis'] = analysis if len(analysis) > 0 else []
    return jsonify(news)

if __name__ == '__main__':
    app.run(debug=True)

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from textblob import TextBlob
import numpy as np
from SportsScrapper import BCCI_Scrapper, ICC_Scrapper, Indian_Athletes_Scrapper


class Validator:
    def __init__(self):
        self.sentiment_pipeline = pipeline('sentiment-analysis')
        self.political_influence_model = pipeline(
            'text-classification', model='typeform/distilbert-base-uncased-mnli')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def bert_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt',
                                max_length=512, truncation=True, padding=True)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embeddings

    def search_official(self, query, player_type, player_platform, search_type):
        data = []
        if search_type == 'bcci':
            scrapper = BCCI_Scrapper()
            response = scrapper.get_player_data(
                query, player_platform, player_type)
            data.extend(response['Response'])
        elif search_type == 'icc':
            scrapper = ICC_Scrapper()
            response = scrapper.get_player_data(query)
            data.extend(response['Response'])
        elif search_type == 'indian_athletes':
            scrapper = Indian_Athletes_Scrapper()
            response = scrapper.get_player_data(query)
            data.extend(response['Response'])
        else:
            scrapper = BCCI_Scrapper()
            response = scrapper.get_player_data(
                query, player_platform, player_type)
            data.extend(response['Response'])
            scrapper = ICC_Scrapper()
            response = scrapper.get_player_data(query)
            data.extend(response['Response'])
            scrapper = Indian_Athletes_Scrapper()
            response = scrapper.get_player_data(query)
            data.extend(response['Response'])
        return data

    def search_unofficial(self, query):
        query = query.lower()
        query = query.replace(' ', '-')
        url = 'https://newsapi.org/v2/everything?'
        parameters = {
            'q': query,
            'apiKey': '399a3fe0b00b4bbfa2188e79abdc5b8b',
            'sources': 'the-times-of-india,the-hindu,hindustan-times,the-indian-express,news18,ndtv,india-today,zee-news,abp-news,india-tv,republic-world,the-quint,the-wire,scroll,the-print',
        }
        response = requests.get(url, params=parameters)
        data = response.json()
        return data['articles']

    def assess_truth(self, unofficial_data, official_data):
        truth_values = []
        vectorizer = TfidfVectorizer()
        for article in unofficial_data:
            unofficial_text = f"{article['title']} {article['description']}"
            similarity_scores = []

            for official_article in official_data:
                official_text = f"{official_article['title']} {official_article['player_name']}"
                vectors = vectorizer.fit_transform(
                    [official_text, unofficial_text])
                similarity = cosine_similarity(
                    vectors[0:1], vectors[1:2])[0][0]
                similarity_scores.append(similarity)
            truth_value = max(similarity_scores)
            if article['source']['name'] in ['The Times of India', 'The Hindu', 'Hindustan Times', 'The Indian Express', 'News18', 'NDTV', 'India Today', 'Zee News', 'ABP News', 'India TV', 'Republic World', 'The Quint', 'The Wire', 'Scroll', 'The Print']:
                truth_value = min(truth_value + 0.4, 1)
            truth_values.append(truth_value)
        return truth_values

    def detect_influence(self, article):
        content = article['content']
        sentiment_result = self.sentiment_pipeline(content)
        emotional_influence = sentiment_result[0]['label'] in [
            'NEGATIVE', 'POSITIVE']
        political_result = self.political_influence_model(content)
        political_influence = any(
            label['label'] == 'POLITICS' and label['score'] > 0.5 for label in political_result)
        return political_influence, emotional_influence

    def cluster_articles(self, articles):
        contents = [article['content'] for article in articles]
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(contents)

        true_k = 5
        model = KMeans(n_clusters=true_k, random_state=42)
        model.fit(X)

        labels = model.labels_
        cluster_dict = {i: [] for i in range(true_k)}
        for idx, label in enumerate(labels):
            cluster_dict[label].append(articles[idx])

        return cluster_dict

    def sentiment_analysis(self, article):
        analysis = TextBlob(article['content'])
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity
        if polarity > 0:
            sentiment = 'Positive, which means the text is expressing positive emotions or opinions'
        elif polarity < 0:
            sentiment = 'Negative, which means the text is expressing negative emotions or opinions'
        else:
            sentiment = 'Neutral, which means the text is neither positive nor negative'
        if subjectivity >= 0.5:
            objectivity = 'Subjective, which means the text is based on opinions or beliefs'
        else:
            objectivity = 'Objective, which means the text is based on facts or evidence'

        return {
            'polarity': polarity,
            'polarity_label': sentiment,
            'subjectivity': subjectivity,
            'subjectivity_label': objectivity
        }

    def relevance_score(self, article, query):
        title = article['title']
        description = article['description']
        content = article['content']
        combined_text = f"{title} {description} {content}"
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([query, combined_text])
        tfidf_similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        query_embedding = self.bert_embedding(query)
        text_embedding = self.bert_embedding(combined_text)
        bert_similarity = cosine_similarity(
            query_embedding, text_embedding)[0][0]
        return {
            'tfidf_similarity': tfidf_similarity,
            'bert_similarity': bert_similarity
        }

    def search(self, query, player_type, player_platform, type):
        official_data = self.search_official(
            query, player_type, player_platform, type)
        unofficial_data = self.search_unofficial(query)

        truth_values = self.assess_truth(unofficial_data, official_data)
        influences = [self.detect_influence(article)
                      for article in unofficial_data]
        clustered_articles = self.cluster_articles(unofficial_data)
        sentiments = [self.sentiment_analysis(article)
                      for article in unofficial_data]
        relevance_scores = [self.relevance_score(article, query)
                            for article in unofficial_data]

        for i, article in enumerate(unofficial_data):
            article['truth_value'] = truth_values[i]
            article['political_influence'], article['emotional_influence'] = influences[i]
            article['sentiment_polarity'] = sentiments[i]['polarity']
            article['sentiment_polarity_label'] = sentiments[i]['polarity_label']
            article['sentiment_subjectivity'] = sentiments[i]['subjectivity']
            article['sentiment_subjectivity_label'] = sentiments[i]['subjectivity_label']
            article['relevance_score_tfidf'] = relevance_scores[i]['tfidf_similarity']
            article['relevance_score_bert'] = relevance_scores[i]['bert_similarity']

        unofficial_data = sorted(
            unofficial_data, key=lambda x: x['truth_value'], reverse=True)

        result = {
            'official_data': official_data,
            'unofficial_data': unofficial_data,
            'clusters': clustered_articles,
        }
        return result

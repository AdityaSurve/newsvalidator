import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import numpy as np
from FinanceScrapper import Investopedia_Scrapper
from unidecode import unidecode
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
import torch

class Validator:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.sentiment_pipeline = pipeline("sentiment-analysis")
        self.emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
        political_model_name = "premsa/political-bias-prediction-allsides-mDeBERTa"
        self.political_model = AutoModelForSequenceClassification.from_pretrained(political_model_name)
        self.political_tokenizer = AutoTokenizer.from_pretrained(political_model_name)

    def bert_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt',
                                max_length=512, truncation=True, padding=True)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embeddings

    def search_official(self, query):
        data = []
        scrapper = Investopedia_Scrapper()
        response = scrapper.get_query_data(query)
        data.extend(response['Response'])
        return data

    def search_unofficial(self, query):
        def preprocess_data(data):
            for article in data:
                if article['categories'] != None:
                    categories = article['categories']
                    category_data = []
                    for category in categories:
                        category_data.append(category['name'])
                    article['categories'] = ', '.join(category_data)
                    if not any(category['name'].isalnum() for category in categories):
                        article['categories'] = '-'
                else :
                    article['categories'] = '-'

                if 'claim' in article:
                    del article['claim']
                    
                if 'clusterId' in article:
                    del article['clusterId']

                if 'companies' in article and article['companies']:
                    companies = article['companies']
                    company_data = [company['name'] for company in companies if 'name' in company]
                    article['companies'] = ', '.join(company_data)
                else:
                    article['companies'] = '-'

                if 'entities' in article and article['entities']:
                    entities = article['entities']
                    entity_data = []
                    entity_mentions = []
                    for entity in entities:
                        entity_data.append(entity['data'])
                        entity_mentions.append(entity['mentions'])
                    article['entities'] = ', '.join(
                        [f'{data} : {mentions}' for data, mentions in zip(entity_data, entity_mentions)])
                else:
                    article['entities'] = '-'

                if 'keywords' in article and article['keywords']:
                    keywords = article['keywords']
                    keyword_data = []
                    keyword_mentions = []
                    for keyword in keywords:
                        keyword_data.append(keyword['name'])
                        keyword_mentions.append(keyword['weight'])
                    article['keywords'] = ', '.join(
                        [f'{data} : {mentions}' for data, mentions in zip(keyword_data, keyword_mentions)])
                else:
                    article['keywords'] = '-'

                if 'language' in article:
                    del article['language']

                if 'labels' in article:
                    del article['labels']

                if 'locations' in article:
                    del article['locations']

                if 'links' in article and article['links']:
                    article['links'] = ', '.join(article['links'])
                else:
                    article['links'] = '-'

                if 'refreshDate' in article:
                    del article['refreshDate']

                if 'places' in article:
                    del article['places']

                if 'people' in article:
                    del article['people']

                if 'matchedAuthors' in article:
                    del article['matchedAuthors']

                if 'reprint' in article:
                    del article['reprint']

                if 'reprintGroupId' in article:
                    del article['reprintGroupId']

                if 'translatedDescription' in article:
                    del article['translatedDescription']

                if 'translatedTitle' in article:
                    del article['translatedTitle']

                if 'translatedSummary' in article:
                    del article['translatedSummary']

                if 'translation' in article:
                    del article['translation']

                if 'verdict' in article:
                    del article['verdict']
                
                if article['addDate'] != None:
                    article['addDate'] = article['addDate'].split('T')[0]
                    article['addDate'] = article['addDate'].split('-')
                    article['addDate'] = f"{article['addDate'][2]}/{article['addDate'][1]}/{article['addDate'][0]}"
                else:
                    article['addDate'] = '-'

                if 'authorsByline' in article:
                    article['author'] = article['authorsByline']
                    if not article['author'] or not any(char.isalnum() for char in article['author']):
                        article['author'] = '-'
                    del article['authorsByline']
                else:
                    article['author'] = '-'

                if article['pubDate'] != None:
                    article['pubDate'] = article['pubDate'].split('T')[0]
                    article['pubDate'] = article['pubDate'].split('-')
                    article['pubDate'] = f"{article['pubDate'][2]}/{article['pubDate'][1]}/{article['pubDate'][0]}" 
                else:
                    article['pubDate'] = '-'
                
                article['source'] = article['source']['domain'] 
                
                if 'topics' in article and article['topics'] != None and len(article['topics']) > 0:      
                    topics = article['topics']
                    topic_data = []
                    for topic in topics:
                        topic_data.append(topic['name'])
                    article['topics'] = ', '.join(topic_data)
                else:
                    article['topics'] = '-'

                if 'content' in article and article['content'] != None:
                    if not any(char.isalnum() for char in article['content']):
                        article['content'] = '-'
                else:
                    article['content'] = '-'

            return data
        query = query.lower()
        pages = [0, 1, 2, 3, 4, 5]
        url = 'https://api.goperigon.com/v1/all'
        data = []
        for page in pages:
            try:
                parameters = {
                    'q': query,
                    'apiKey': '20504987-ba9f-48f7-afc1-1fb20ce0849d',
                    'page': page,
                    'language': 'en',
                }
                response = requests.get(url, params=parameters)
                data.extend(response.json()['articles'])
            except:
                pass
        data = preprocess_data(data)
        return data

    def assess_truth(self, unofficial_data, official_data):
        official_embeddings = []
        for official_article in official_data:
            title = unidecode(official_article['title'])
            content = unidecode(official_article['content'])
            description = unidecode(official_article['description'])
            official_text = f"{title} {content} {description}"
            official_embeddings.append(self.bert_embedding(official_text).squeeze())
        truth_values = []
        unofficial_embeddings = []
        for article in unofficial_data:
            unofficial_text = f"{article['title']} {article['description']} {article['content']} {article['summary']}"
            unofficial_embeddings.append(self.bert_embedding(unofficial_text).squeeze())
        official_embeddings = np.array(official_embeddings)
        unofficial_embeddings = np.array(unofficial_embeddings)
        for unofficial_embedding in unofficial_embeddings:
            similarity_scores = cosine_similarity([unofficial_embedding], official_embeddings)[0]
            truth_value = max(similarity_scores)
            truth_values.append(truth_value)
        unofficial_similarity_matrix = cosine_similarity(unofficial_embeddings)
        consensus_scores = unofficial_similarity_matrix.mean(axis=1)
        combined_scores = [(truth_value + consensus_score) / 2 for truth_value, consensus_score in zip(truth_values, consensus_scores)]

        return combined_scores

    def detect_influence(self, article):
        content = article['content']
        sentiment_result = self.sentiment_pipeline(content)
        emotional_influence = sentiment_result[0]['label'] in ['NEGATIVE', 'POSITIVE']
        political_inputs = self.political_tokenizer.encode_plus(content, return_tensors='pt', truncation=True, padding=True)
        political_output = self.political_model(**political_inputs)
        political_probabilities = torch.nn.functional.softmax(political_output.logits, dim=-1)
        political_labels = ['Neutral', 'Liberal', 'Conservative']
        political_scores = political_probabilities.detach().numpy()[0]
        political_influence = any(score > 0.5 for score in political_scores if political_labels[political_scores.tolist().index(score)] != 'Neutral')
        
        return political_influence, emotional_influence

    def sentiment_analysis(self, article):
        if 'content' in article and article['content'] is not None:
            text = article['content']
            analysis = TextBlob(text)
            subjectivity = analysis.sentiment.subjectivity
            if 'sentiment' not in article or article['sentiment'] is None:
                article['sentiment'] = None
            if article['sentiment'] is None:
                model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                inputs = tokenizer.encode_plus(text, return_tensors='pt', truncation=True, padding=True)
                output = model(**inputs)
                probabilities = F.softmax(output.logits, dim=-1)
                sentiment_values = probabilities.detach().numpy()[0]
                article['sentiment'] = {}
                article['sentiment']['negative'] = round(float(sentiment_values[0]), 6)
                article['sentiment']['neutral'] = round(float(sentiment_values[1]), 6)
                article['sentiment']['positive'] =  round(float(sentiment_values[2]), 6)
            if subjectivity >= 0.5:
                objectivity = 'Subjective'
            else:
                objectivity = 'Objective'
            sentiment_polarity_labels = ['Negative', 'Neutral', 'Positive']
            sentiment_polarity_scores = [article['sentiment']['negative'], article['sentiment']['neutral'], article['sentiment']['positive']]
            max_sentiment_polarity_score = max(sentiment_polarity_scores)
            sentiment_polarity_label = sentiment_polarity_labels[sentiment_polarity_scores.index(max_sentiment_polarity_score)]
            return {
                'polarity_label': sentiment_polarity_label,
                'subjectivity': subjectivity,
                'subjectivity_label': objectivity
            }
        else:
            return None

    def relevance_score(self, article, query):
        # def preprocess_text(text):
        #     if text is None:
        #         return ''
        #     lemmatizer = WordNetLemmatizer()
        #     text = text.lower()
        #     text = "".join(
        #         [char for char in text if char not in string.punctuation])
        #     words = word_tokenize(text)
        #     words = [lemmatizer.lemmatize(word)
        #              for word in words if word not in ENGLISH_STOP_WORDS]
        #     return " ".join(words)

        # def get_word2vec_embeddings(text):
        #     words = word_tokenize(text)
        #     model = Word2Vec([words], min_count=1)
        #     embeddings = np.mean([model.wv[word] for word in words], axis=0)
        #     return embeddings.reshape(1, -1)

        # title = preprocess_text(article['title'])
        # description = preprocess_text(article['description'])
        # content = preprocess_text(article['content'])
        # combined_text = f"{title} {description} {content}"
        # query = preprocess_text(query)

        # vectorizer = TfidfVectorizer()
        # vectors = vectorizer.fit_transform([query, combined_text])
        # tfidf_similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

        # query_embedding = self.bert_embedding(query)
        # text_embedding = self.bert_embedding(combined_text)
        # bert_similarity = cosine_similarity(
        #     query_embedding, text_embedding)[0][0]

        # query_embedding_w2v = get_word2vec_embeddings(query)
        # text_embedding_w2v = get_word2vec_embeddings(combined_text)
        # w2v_similarity = cosine_similarity(
        #     query_embedding_w2v, text_embedding_w2v)[0][0]

        # similarity = np.average(
        #     [bert_similarity, tfidf_similarity, w2v_similarity], weights=[0.5, 0.3, 0.2])

        # return {
        #     'similarity': similarity
        # }
        article_text = f"{article['title']} {article['description']} {article['content']} {article['summary']}"
        article_embedding = self.bert_embedding(article_text).squeeze()
        query_embedding = self.bert_embedding(query).squeeze()
        similarity = cosine_similarity([article_embedding], [query_embedding])[0][0]
        return {
            'similarity': similarity
        }

    def search(self, query):
        official_data = self.search_official(query)
        entity = query.split(' ')[0]
        unofficial_data = self.search_unofficial(entity)
        if isinstance(official_data, str):
            official_data = [json.loads(official_data)]
        if isinstance(unofficial_data, str):
            unofficial_data = [json.loads(unofficial_data)]

        truth_values = self.assess_truth(unofficial_data, official_data)
        influences = [self.detect_influence(article)
                      for article in unofficial_data]
        sentiments = [self.sentiment_analysis(article)
                      for article in unofficial_data]
        relevance_scores = [self.relevance_score(article, query)
                            for article in unofficial_data]

        for i, article in enumerate(unofficial_data):
            article['truth_value'] = truth_values[i]
            article['political_influence'], article['emotional_influence'] = influences[i]
            article['sentiment_polarity_label'] = sentiments[i]['polarity_label']
            article['sentiment_subjectivity'] = sentiments[i]['subjectivity']
            article['sentiment_subjectivity_label'] = sentiments[i]['subjectivity_label']
            article['relevance_score'] = relevance_scores[i]['similarity']

        unofficial_data = sorted(
            unofficial_data, key=lambda x: x['score'], reverse=True)

        unofficial_count = len(unofficial_data)
        official_count = len(official_data)

        if official_count > 0 or unofficial_count > 0:
            result = {
                'status': 'success',
                'unofficial_count': unofficial_count,
                'official_count': official_count,
                'official_data': official_data,
                'unofficial_data': unofficial_data,
            }
        else:
            result = {
                'status': 'error',
                'message': 'No data found'
            }
        return result

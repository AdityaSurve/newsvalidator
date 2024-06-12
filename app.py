import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

st.title('News Data')

with st.form(key='my_form'):
    player_name = st.text_input(label='Enter query ... ')
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    response = requests.get(
        f'http://localhost:5000/validator?player_name={player_name}&platform=international&type=men&news_type=bcci')

    if response.status_code == 200:
        response_data = response.json()
        official_data = pd.DataFrame(response_data['official_data'])
        unofficial_data = pd.DataFrame(response_data['unofficial_data'])
        st.set_option('deprecation.showPyplotGlobalUse', False)

        st.write(f'Official Data : ' + str(len(official_data)) + ' articles')
        st.dataframe(official_data)

        st.write(f'Unofficial Data : ' +
                 str(len(unofficial_data)) + ' articles')
        st.dataframe(unofficial_data)

        unofficial_data['source'] = unofficial_data['source'].apply(
            lambda x: x['name'])
        source_counts = unofficial_data['source'].value_counts()
        st.write('Out of ' + str(len(unofficial_data)) + ' unofficial articles from ' +
                 str(len(source_counts)) + ' sources, the top 6 sources are: ' + ', '.join(source_counts.index[:6]))
        if len(source_counts) > 6:
            source_counts = source_counts[:6]
            source_counts['Others'] = unofficial_data['source'].value_counts(
            )[6:].sum()

        plt.figure(figsize=(10, 10))
        plt.pie(source_counts, labels=source_counts.index, autopct='%1.1f%%')
        plt.title('Number of articles per source')
        st.pyplot()

        source_truth_values = unofficial_data.groupby(
            'source')['truth_value'].agg(['max', 'mean', 'min']).reset_index()
        source_truth_values['source_index'] = range(len(source_truth_values))
        source_truth_values_with_index = source_truth_values.copy()
        source_truth_values.set_index('source_index', inplace=True)

        source_truth_values[['max', 'mean', 'min']].plot(
            kind='line', figsize=(10, 10))
        plt.title('Truth value per source')
        plt.xlabel('Source Index')
        plt.ylabel('Truth value')
        st.pyplot()
        st.table(source_truth_values_with_index[['source']])
        st.write('Truth value is a measure of the reliability of the news source. It ranges from 0 to 1, where 0 means the news source is unreliable and 1 means the news source is reliable.')

        source_relevance_scores = unofficial_data.groupby(
            'source')['relevance_score'].agg(['max', 'mean', 'min']).reset_index()
        source_relevance_scores['source_index'] = range(
            len(source_relevance_scores))
        source_relevance_scores_with_index = source_relevance_scores.copy()
        source_relevance_scores.set_index('source_index', inplace=True)

        source_relevance_scores[['max', 'mean', 'min']].plot(
            kind='line', figsize=(10, 10))
        plt.title('Relevance score per source')
        plt.xlabel('Source Index')
        plt.ylabel('Relevance score')
        st.pyplot()
        st.table(source_relevance_scores_with_index[['source']])
        st.write('Relevance score is a measure of how relevant the news article is to the query. It ranges from 0 to 1, where 0 means the news article is irrelevant and 1 means the news article is relevant.')

        sentiment_counts = unofficial_data['sentiment_polarity_label'].value_counts(
        )
        labels = [label.split(',')[0] for label in sentiment_counts.index]
        plt.figure(figsize=(10, 10))
        plt.bar(range(len(labels)), sentiment_counts)
        plt.title('Number of articles per sentiment polarity label')
        plt.xlabel('Sentiment polarity label')
        plt.xticks(range(len(labels)), labels)
        plt.ylabel('Number of articles')
        for i, count in enumerate(sentiment_counts):
            labels[i] += f' ({count})'
        plt.legend(labels)

        st.pyplot()
        st.write('Positive : means the sentiment is positive, Negative : means the sentiment is negative, Neutral : means the sentiment is neutral')

        subjectivity_counts = unofficial_data['sentiment_subjectivity_label'].value_counts(
        )
        labels = [label.split(',')[0] for label in subjectivity_counts.index]
        plt.figure(figsize=(10, 10))
        plt.bar(range(len(labels)), subjectivity_counts)
        plt.title('Number of articles per sentiment subjectivity label')
        plt.xlabel('Sentiment subjectivity label')
        plt.xticks(range(len(labels)), labels)
        plt.ylabel('Number of articles')
        for i, count in enumerate(subjectivity_counts):
            labels[i] += f' ({count})'
        st.pyplot()
        st.write('Subjective : means the text is based on opinions or beliefs, Objective : means the text is based on facts or evidence')

    else:
        st.write('Error fetching data')

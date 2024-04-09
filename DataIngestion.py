import requests

class NewsApi(object):
    def __init__(self):
        self.api_key = '5ade57d71e614bbeb04fb1f8d3d3b70b'

    def get_news(self, category=None, source=None, query=None, language=None, country=None, date=None, start_date=None, end_date=None):
        try:
            url = 'https://newsapi.org/v2/everything?apiKey=' + self.api_key
            if category:
                url += '&category=' + category
            if source:
                url += '&sources=' + source
            if query:
                url += '&q=' + query
            if language:
                url += '&language=' + language
            if country:
                url += '&country=' + country
            if date:
                url += '&from=' + date
            if start_date and end_date:
                url += '&from=' + start_date + '&to=' + end_date
            response = requests.get(url)
            return response.json()
        except Exception as e:
            return {'error': str(e)}
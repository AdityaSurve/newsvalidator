import requests
from bs4 import BeautifulSoup
import pandas as pd
import os


class ICAR_Scrapper:
    def __init__(self):
        self.url = "https://www.icar.org.in/search/node?keys="
        self.query = None

    def get_query_data(self, query):
        processed_query = query.replace(" ", "%20")

        if not processed_query:
            return {"Error": "Invalid input"}

        self.query = processed_query

        url = self.url + processed_query

        response = self.get_data(processed_query, url)

        if response:
            return {"Response": response}
        else:
            return {"Error": "No data found"}

    def get_data(self, query, url):
        try:
            response = requests.get(url, verify=False)
            response.raise_for_status()
        except requests.RequestException as e:
            return {"Error": str(e)}

        soup = BeautifulSoup(response.text, 'html.parser')
        data = []

        query_data = soup.find(
            'ol', class_='search-results node_search-results')
        if query_data:
            title = ""
            date = ""
            content = ""
            link = ""
            place = ""

            li_tags = query_data.find_all('li')
            if li_tags:
                for li in li_tags:
                    title_holder = li.find('h3')
                    title_holder = title_holder.find('a')
                    link = title_holder['href']
                    title = title_holder.text

                    home_page = requests.get(link, verify=False)
                    home_page.raise_for_status()
                    home_soup = BeautifulSoup(home_page.text, 'html.parser')

                    content_holder = home_soup.find(
                        'div', class_='node__content clearfix')
                    if content_holder:
                        content_holder = content_holder.find(
                            'div', class_='clearfix text-formatted field field--name-body field--type-text-with-summary field--label-hidden field__item')

                    if content_holder:
                        paragraphs = content_holder.find_all('p')
                        if paragraphs[0].find('em'):
                            date_place = paragraphs[0].find('em')
                            date_place = date_place.text
                            date_place = date_place.split(',')
                            if len(date_place) == 2:
                                if any(char.isdigit() for char in date_place[0]) and any(char.isdigit() for char in date_place[1]):
                                    date = date_place[0] + ', ' + date_place[1]
                                    place = "-"
                                else:
                                    if any(char.isdigit() for char in date_place[0]):
                                        date = date_place[0]
                                        place = date_place[1]
                                    else:
                                        date = date_place[1]
                                        place = date_place[0]
                            elif len(date_place) == 3:
                                date = date_place[0] + ', ' + date_place[1]
                                place = date_place[2]
                            content = ""
                            for i in range(1, len(paragraphs)):
                                content += paragraphs[i].text + " "
                        else:
                            date = "-"
                            place = "-"
                            content = ""
                            for i in range(0, len(paragraphs)):
                                content += paragraphs[i].text + " "

                    data.append({
                        "title": title,
                        "date": date,
                        "content": content,
                        "link": link,
                        "place": place,
                        "query": query,
                        "category": "Agriculture"
                    })

            if data:
                df = pd.DataFrame(data)
                header = ['title', 'date', 'content',
                          'link', 'place', 'query', 'category']
                if os.path.isfile('ICAR.csv') and os.path.getsize('ICAR.csv') > 0:
                    df.to_csv('ICAR.csv', mode='a', header=False, index=False)
                else:
                    df.to_csv('ICAR.csv', mode='a', header=header, index=False)
                return data
            else:
                return {"Error": "No data found"}
        else:
            return {"Error": "No data found"}


if __name__ == "__main__":
    scrapper = ICAR_Scrapper()

    query = "Agriculture"
    scrapped_data = scrapper.get_query_data(query)

    query = "Mango"
    scrapped_data = scrapper.get_query_data(query)

    query = "Wheat"
    scrapped_data = scrapper.get_query_data(query)

    query = "Rice"
    scrapped_data = scrapper.get_query_data(query)

    query = "Cotton"
    scrapped_data = scrapper.get_query_data(query)

    query = "Pulses"
    scrapped_data = scrapper.get_query_data(query)

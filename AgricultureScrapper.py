import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from selenium import webdriver

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


class FAO_Scrapper:
    def __init__(self):
        self.url = "https://www.fao.org/home/search/en/?"
        self.query = None

    def get_query_data(self, query):
        processed_query = query.replace(" ", "+")

        if not processed_query:
            return {"Error": "Invalid input"}

        self.query = processed_query

        url = self.url + "q=" + processed_query
        response = self.get_data(processed_query, url)

        if response:
            return {"Response": response}
        else:
            return {"Error": "No data found"}

    def get_data(self, query, url):
        try:
            driver = webdriver.Edge()
            driver.get(url)
        except requests.RequestException as e:
            return {"Error": str(e)}

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        data = []

        query_data = soup.find('div', class_='gsc-expansionArea')
        if query_data:
            cards = query_data.find_all(
                'div', class_='gsc-webResult gsc-result')
            for card in cards:
                title = ""
                content = ""
                description = ""
                link = ""
                category = "Agriculture"

                card = card.find('div', class_='gs-webResult gs-result')
                header = card.find('div', class_='gsc-thumbnail-inside')
                title_holder = header.find('div', class_='gs-title')
                a_tag = title_holder.find('a')
                link = a_tag['href']
                title = a_tag.text

                home_driver = webdriver.Edge()
                home_driver.get(link)
                home_soup = BeautifulSoup(
                    home_driver.page_source, 'html.parser')

                content_holder = home_soup.find('section', id='content')
                if content_holder:
                    content_holder = content_holder.find(
                        'div', class_='main-internal')
                    if content_holder:
                        desc_holder = content_holder.find(
                            'div', class_='csc-default')
                        if desc_holder:
                            desc_holder = desc_holder.find(
                                'p', class_='bodytext')
                            if desc_holder:
                                description = desc_holder.text
                            else:
                                description = "-"

                        content_holder = content_holder.find_all(
                            'div', class_='rgaccord1-nest')
                        if content_holder:
                            content = ""
                            for div in content_holder:
                                section_title_holder = div.find('h3')
                                section_title = section_title_holder.text

                                text_holder = div.find('div')
                                text_holder = text_holder.find('div')
                                text = ''

                                if text_holder:
                                    text = text_holder.text

                                content += section_title + "\n" + text + "\n\n"

                data.append({
                    "title": title,
                    "content": content,
                    "description": description,
                    "link": link,
                    "query": query,
                    "category": category
                })
            home_driver.quit()
            driver.quit()
            if data:
                df = pd.DataFrame(data)
                header = ['title', 'content', 'description',
                          'link', 'query', 'category']
                if os.path.isfile('FAO.csv') and os.path.getsize('FAO.csv') > 0:
                    df.to_csv('FAO.csv', mode='a', header=False, index=False)
                else:
                    df.to_csv('FAO.csv', mode='a', header=header, index=False)
                return data
            else:
                return {"Error": "No data found"}

        else:
            return {"Error": "No data found"}

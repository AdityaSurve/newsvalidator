import requests
from bs4 import BeautifulSoup
import pandas as pd
import os


class Investopedia_Scrapper:
    def __init__(self):
        self.url = "https://www.investopedia.com/search?"
        self.query = None
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

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
            response = self.session.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            return {"Error": str(e)}

        soup = BeautifulSoup(response.text, 'html.parser')
        data = []

        query_data = soup.find(
            'div', class_='comp search-results__results mntl-block')
        if query_data:
            cards = query_data.find_all(
                'div', class_='comp search-results__list mntl-block')
            for card in cards:
                title = ""
                desc = ""
                date = ""
                author = ""
                author_link = ""
                reviewer = ""
                reviewer_link = ""
                fact_checker = ""
                fact_checker_link = ""
                content = ""
                link = ""
                keypoints = ""
                category = "Finance"

                link_holder = card.find('a')
                link = link_holder['href']
                title_holder = link_holder.find('h3')
                title = title_holder.text
                desc_holder = card.find(
                    'div', class_='comp search-results__description mntl-text-block')
                desc = desc_holder.text

                try:
                    home_page = self.session.get(link)
                    home_page.raise_for_status()
                    home_soup = BeautifulSoup(home_page.text, 'html.parser')

                    base_content_holder = home_soup.find(
                        'div', class_='loc article-content')
                    if base_content_holder:
                        sec_content_holder = base_content_holder.find(
                            'div', class_='comp article-body mntl-block')
                        if sec_content_holder:
                            content_holder = sec_content_holder.find(
                                'div', class_='comp mntl-sc-page mntl-block article-body-content')
                            if content_holder:
                                paragraphs = content_holder.find_all('p')
                                content = ""
                                for p in paragraphs:
                                    content += p.text + " "
                                main_keypoints = content_holder.find(
                                    'div', class_='comp mntl-sc-block finance-sc-block-callout mntl-block')
                                if main_keypoints:
                                    sec_keypoints = main_keypoints.find(
                                        'div', class_='comp mntl-sc-block mntl-sc-block-callout mntl-block theme-whatyouneedtoknow')
                                    if sec_keypoints:
                                        tert_keypoints = sec_keypoints.find(
                                            'div', class_='comp mntl-sc-block-callout-body mntl-text-block')
                                        if tert_keypoints:
                                            quad_keypoints = tert_keypoints.find(
                                                'ul')
                                            if quad_keypoints:
                                                keypoints = quad_keypoints.find_all(
                                                    'li')
                                                if keypoints:
                                                    keys = ""
                                                    for key in keypoints:
                                                        keys += "* " + key.text + " "
                                                    keypoints = keys
                    main_header_holder = home_soup.find(
                        'div', class_='loc article-pre-content')
                    if main_header_holder:
                        header_holder = main_header_holder.find(
                            'header', class_='comp article-header mntl-block right-rail__offset js-toc-appear')
                        if header_holder:
                            base_meta_holder = header_holder.find(
                                'div', class_='comp article-meta mntl-block')
                            if base_meta_holder:
                                meta_holder = base_meta_holder.find(
                                    'div', class_='comp finance-bylines mntl-bylines')
                                if meta_holder:
                                    base_author_and_date = meta_holder.find(
                                        'div', class_='comp mntl-bylines__group mntl-block mntl-bylines__group--author')
                                    if base_author_and_date:
                                        base_author_holder = base_author_and_date.find(
                                            'div', class_='comp mntl-bylines__item mntl-attribution__item mntl-attribution__item--has-date')
                                        if base_author_holder:
                                            sec_author_holder = base_author_holder.find(
                                                'div')
                                            if sec_author_holder:
                                                author_holder = sec_author_holder.find(
                                                    'a')
                                                if author_holder:
                                                    author = author_holder.text
                                                    author_link = author_holder['href']
                                        date_holder = base_author_and_date.find(
                                            'div', class_='mntl-attribution__item-date')
                                        if date_holder:
                                            date = date_holder.text
                                    base_reviewer_holder = meta_holder.find(
                                        'div', class_='comp mntl-bylines__group mntl-block mntl-bylines__group--finance_reviewer')
                                    if base_reviewer_holder:
                                        sec_reviewer_holder = base_reviewer_holder.find(
                                            'div', class_='comp mntl-bylines__item mntl-attribution__item')
                                        if sec_reviewer_holder:
                                            tert_reviewer_holder = sec_reviewer_holder.find(
                                                'div')
                                            if tert_reviewer_holder:
                                                quad_reviewer_holder = tert_reviewer_holder.find(
                                                    'a')
                                                if quad_reviewer_holder:
                                                    reviewer = quad_reviewer_holder.text
                                                    reviewer_link = quad_reviewer_holder['href']
                                    base_fact_checker_holder = meta_holder.find(
                                        'div', class_='comp mntl-bylines__group mntl-block mntl-bylines__group--fact_checker')
                                    if base_fact_checker_holder:
                                        sec_fact_checker_holder = base_fact_checker_holder.find(
                                            'div', class_='comp mntl-bylines__item mntl-attribution__item')
                                        if sec_fact_checker_holder:
                                            tert_fact_checker_holder = sec_fact_checker_holder.find(
                                                'div')
                                            if tert_fact_checker_holder:
                                                quad_fact_checker_holder = tert_fact_checker_holder.find(
                                                    'a')
                                                if quad_fact_checker_holder:
                                                    fact_checker = quad_fact_checker_holder.text
                                                    fact_checker_link = quad_fact_checker_holder['href']
                except requests.RequestException as e:
                    return {"Error": str(e)}

                data.append({
                    "title": title,
                    "description": desc,
                    "date": date,
                    "author": author,
                    "author_link": author_link,
                    "reviewer": reviewer,
                    "reviewer_link": reviewer_link,
                    "fact_checker": fact_checker,
                    "fact_checker_link": fact_checker_link,
                    "content": content,
                    "link": link,
                    "keypoints": keypoints,
                    "category": category
                })

            if data:
                return data
            else:
                return {"Error": "No data found"}
        else:
            return {"Error": "No data found"}

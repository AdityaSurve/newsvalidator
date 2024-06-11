from bs4 import BeautifulSoup
import requests
import pandas as pd
import os


class BCCI_Scrapper:
    def __init__(self):
        self.url = "https://www.bcci.tv/search?"
        self.player_name = None
        self.platform = None
        self.type = None

    def get_player_data(self, player_name, platform, type):
        processed_player_name = player_name.replace(" ", "+")
        processed_platform = platform.lower()
        processed_type = type.lower()

        if not processed_platform or not processed_type or not processed_player_name:
            return {"Error": "Invalid input"}

        self.player_name = processed_player_name
        self.platform = processed_platform
        self.type = processed_type

        url = self.url + "platform=" + self.platform + "&type=" + \
            self.type + "&term=" + self.player_name + "&content_type=all"

        response = self.get_data(
            url, processed_platform, processed_type, processed_player_name)

        if response:
            return {"Response": response}
        else:
            return {"Error": "No data found"}

    def get_data(self, url, processed_platform, processed_type, player_name):
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            print(e)

        soup = BeautifulSoup(response.text, 'html.parser')
        data = []

        player_data = soup.find(
            'div', class_='lv pc video-section-append-video-here')
        if player_data:
            player_card = player_data.find_all(
                'div', class_='slick-card m-0 lv-bg hoverVideoPlayNow')
            for card in player_card:
                header_content = card.find('div', class_='bottom')
                title_holder = None
                title = ""
                date = ""
                views = ""
                image_url = ""
                link = ""

                if header_content.find('div', class_='text-detail br-b'):
                    title_holder = header_content.find(
                        'div', class_='text-detail br-b')
                    date_holder = header_content.find(
                        'div', class_='text-detail br-b')
                    date = date_holder.find('span').text
                    views_holder = header_content.find(
                        'div', class_='tour-overlay-details')
                    list = views_holder.find('ul')
                    views = list.find_all('li')[0]
                    views = views.find('span', class_="me-3").text
                    views = views.replace("&nbsp;", "").replace(
                        "views", "").replace("\n", "").replace("\xa0", "").strip()
                    if "k" in views:
                        views = views.replace("k", "")
                        views = float(views) * 1000
                    elif "m" in views:
                        views = views.replace("m", "")
                        views = float(views) * 1000000
                    views = int(views)

                    links_holder = card.find('a')
                    link = links_holder['data-share']
                    image_url = links_holder['data-thumbnile']

                elif header_content.find('div', class_='text-detail pb-0'):
                    title_holder = header_content.find(
                        'div', class_='text-detail pb-0')
                    date_holder = header_content.find(
                        'div', class_='tour-overlay-details')
                    ul = date_holder.find('ul')
                    date_ = ul.find_all('li')[0]
                    date = date_.find('span').text
                    views = "-"
                    links_holder = card.find('a')
                    link = links_holder['href']
                    link = "https://www.bcci.tv" + link
                    image_url = links_holder['data-thumbnile']
                if title_holder:
                    title = title_holder.find('p').text
                temp_player_name = player_name.replace("+", " ")
                data.append({
                    "title": title,
                    "date": date,
                    "views": views,
                    "platform": processed_platform,
                    "type": processed_type,
                    "player_name": temp_player_name,
                    "image_url": image_url,
                    "link": link,
                    "sport": "Cricket"
                })
            if data:
                return data
            else:
                return {"Error": "No data found"}
        else:
            return {"Error": "No data found"}


class Indian_Athletes_Scrapper:
    def __init__(self):
        self.url = "https://indianathletics.in/?"
        self.player_name = None

    def get_player_data(self, player_name):
        processed_player_name = player_name.replace(" ", "+")

        if not processed_player_name:
            return {"Error": "Invalid input"}

        self.player_name = processed_player_name

        url = self.url + "s=" + self.player_name

        response = self.get_data(processed_player_name, url)

        if response:
            return {"Response": response}
        else:
            return {"Error": "No data found"}

    def get_data(self, player_name, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            print(e)

        soup = BeautifulSoup(response.text, 'html.parser')
        data = []

        player_data = soup.find('ol', class_='search_res')
        if player_data:
            title = ""
            date = ""
            content = ""
            image_url = ""
            link = ""

            li_tags = player_data.find_all('li')
            for li in li_tags:
                divs_holder = li.find('div')
                title_holder = divs_holder.find('h3')
                title = title_holder.find('a')['title']
                title = title.replace("Permalink to ", "")
                link = title_holder.find('a')['href']

                home_page = requests.get(link)
                home_page.raise_for_status()
                home_soup = BeautifulSoup(home_page.text, 'html.parser')
                content_holder = home_soup.find(
                    'div', class_='post_content col span_3_of_4')
                if content_holder:
                    content_holder = content_holder.find(
                        'div', class_="section")
                    image_holder = content_holder.find(
                        'div', class_='pic')
                    if image_holder:
                        image_holder = image_holder.find('a')
                    if image_holder:
                        image_url = image_holder['href']

                    date_holder = content_holder.find(
                        'p', class_='post_meta')
                    date_span = date_holder.find('span')
                    date = date_span.text

                    content = content_holder.find(
                        'div', class_="post_description")

                    text = ''
                    if content:
                        content = content.find_all('p')
                        for c in content:
                            text += c.text + "\n\n"
                        text = text.strip()
                    temp_player_name = player_name.replace("+", " ")
                    data.append({
                        "title": title,
                        "date": date,
                        "content": text,
                        "player_name": temp_player_name,
                        "image_url": image_url,
                        "link": link,
                        "sport": "Athletics"
                    })
            if data:
                return data
            else:
                return {"Error": "No data found"}
        else:
            return {"Error": "No data found"}


class ICC_Scrapper:
    def __init__(self):
        self.url = "https://www.icc-cricket.com/search?"
        self.player_name = None

    def get_player_data(self, player_name):
        processed_player_name = player_name.replace(" ", "%20")

        if not processed_player_name:
            return {"Error": "Invalid input"}

        self.player_name = processed_player_name

        url = self.url + "q=" + self.player_name

        response = self.get_data(processed_player_name, url)

        if response:
            return {"Response": response}
        else:
            return {"Error": "No data found"}

    def get_data(self, player_name, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            return {"Error": str(e)}

        soup = BeautifulSoup(response.text, 'html.parser')
        data = []

        player_data = soup.find(
            'div', class_='my-4 grid grid-cols-1 lg:grid-cols-4 gap-6 lg:gap-x-6 lg:gap-y-10')
        if player_data:
            cards = player_data.find_all(
                'div', class_='h-[541px] relative rounded-lg lg:rounded-[14px] overflow-hidden')
            for card in cards:
                title = ""
                date = ""
                content = ""
                image_url = ""
                link = ""
                link_holder = card.find('a')
                link = link_holder['href']
                image_holder = link_holder.find_all(
                    'div')[0]
                image_holder = image_holder.find('picture')
                image_holder = image_holder.find('img')
                image_url = image_holder['src']
                divs = link_holder.find_all('div')
                if len(divs) > 1:
                    content_holder = divs[1]
                    content_holder = content_holder.find('div')
                    title_holder = content_holder.find_all('div')[1] if len(
                        content_holder.find_all('div')) > 1 else None
                    if title_holder:
                        title = title_holder.text
                    content_holder = content_holder.find(
                        'div', class_='text-sm font-bold text-white leading-[1.2] lg:text-lg lg:leading-[1.4] lg:-tracking-[0.72px]')
                    if content_holder:
                        content = content_holder.text
                    date_holder = content_holder.find('time')
                    date = date_holder.text if date_holder else ""
                else:
                    continue

                temp_player_name = player_name.replace("%20", " ")
                data.append({
                    "title": title,
                    "date": date,
                    "content": content,
                    "player_name": temp_player_name,
                    "image_url": image_url,
                    "link": link,
                    "sport": "Cricket"
                })

            if data:
                return data
            else:
                return {"Error": "No data found"}
        else:
            return {"Error": "No data found"}


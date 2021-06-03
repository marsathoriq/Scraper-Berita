import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
import requests
from bs4 import BeautifulSoup


def get_html_content(url):
    html_content = requests.get(url).text
    soup = BeautifulSoup(html_content, "lxml")
    return soup


def get_urls_by_date(date):
    all_list_url = []
    list_url = [1]
    page = 1
    while len(list_url) > 0:
        # example https://www.detik.com/search/searchall?query=jakarta&siteid=2&sortby=time&sorttime=0&fromdatex=06/05/2021&todatex=06/05/2021
        url = "https://www.detik.com/search/searchall?query=jakarta&siteid=3&sortby=time&sorttime=0&fromdatex={}&todatex={}&page={}".format(
            date, date, page
        )

        content = get_html_content(url)
        related_data = content.find_all("article")
        list_url = []
        for i in related_data:
            list_url.append(i.find("a").attrs["href"])
        page += 1

        all_list_url = all_list_url + list_url
    all_list_url = [news for news in all_list_url if "/detiktv/" not in news]
    all_list_url = [news for news in all_list_url if "/foto" not in news]
    return all_list_url


def get_news_attr(url):

    news_attr = get_html_content(url)

    # get date
    try:
        datetime_str = news_attr.find(
            "div", attrs={"class": "detail__date"}
        ).text.split(",")[1]
        date_str = " ".join(datetime_str.split()[:3])
    except:
        date_str = ""

    # get title
    try:
        title_content = news_attr.find("h1", attrs={"class": "detail__title"})
        title = " ".join(title_content.text.split())
    except:
        title = ""

    # get image
    try:
        image_content = news_attr.find("div", attrs={"class": "detail__media"})
        link_image = image_content.find("img").attrs["src"]
    except:
        link_image = ""

    # get tags
    try:
        tag_content = news_attr.find("div", attrs={"class": "nav"})
        tags = tag_content.find_all("a", attrs={"class": "nav__item"})
        all_tags = []
        for _, tag in enumerate(tags):
            all_tags = all_tags + [tag.text]
    except:
        all_tags = []

    # get content
    try:
        total_page = len(
            news_attr.find("div", attrs={"class": "detail__long-nav"}).find_all(
                "a", attrs={"class": "detail__anchor-numb"}
            )
        )
    except:
        total_page = 1

    try:
        cnt_paragraph = 0
        news_content = {}
        for i in range(1, total_page + 1):
            if i != 1:
                news_attr = get_html_content(url + "/" + str(i))
            news_texts = news_attr.find(
                "div", attrs={"class": "detail__body-text itp_bodycontent"}
            )
            paragraphs = news_texts.find_all("p")

            id = 0
            cnt = 0
            while id < len(paragraphs):
                # print(paragraphs[id])
                if paragraphs[id].text.lower() in [
                    "simak selengkapnya di halaman selanjutnya.",
                    "\n",
                ]:
                    id += 1
                else:
                    news_content[cnt_paragraph + cnt] = paragraphs[id].text
                    cnt += 1
                    id += 1

            cnt_paragraph += cnt
    except:
        news_content = {}

    return {
        "portal": "detik.com",
        "date": date_str,
        "link": url,
        "title": title,
        "image": link_image,
        "content": news_content,
        "tags": all_tags,
    }


def get_detik_dataframe_from_date(date):
    url_list = get_urls_by_date(date)
    all_data = []
    for url in url_list:
        all_data += [get_news_attr(url)]
        break

    return pd.DataFrame(all_data)

if __name__ == "__main__":
    ## Scrap by date example
    date = "10/05/2021"
    url_list = get_urls_by_date(date)
    all_data = []
    for url in url_list:
        all_data += [get_news_attr(url)]
        break
    pd.DataFrame(all_data).to_csv("data_berita_detik.csv", index=False)

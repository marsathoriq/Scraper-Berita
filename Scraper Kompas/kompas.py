import requests
import unicodedata
from bs4 import BeautifulSoup
from datetime import datetime

URL_TEMPLATE = "https://www.kompas.com/tag/jakarta?sort=desc&page={}"

def get_html_content(url):
  html_content = requests.get(url).text
  soup = BeautifulSoup(html_content, "lxml")
  return soup

def get_n_urls(count):
    pass

def get_urls_by_date(date_str):
    urls = []
    page = 1
    target_date = datetime.strptime(date_str, "%d/%m/%Y")
    is_end = False

    while not is_end:
        soup = get_html_content(URL_TEMPLATE.format(page))
        contents = soup.find_all("div", class_="article__list")  # get article list
        for content in contents:
            news_datetime_str = content.find("div", class_="article__date").text
            news_date_str = news_datetime_str.split(',')[0]
            news_date = datetime.strptime(news_date_str, "%d/%m/%Y")

            ## Compare the current article with target date
            if news_date.date() > target_date.date():
                break
            if news_date.date() < target_date.date():
                is_end = True
                break
            if news_date.date() == target_date.date():
                urls.append(content.find('a').attrs['href'])

        page += 1
        

    return urls

def get_news_attr(url):
    page = get_html_content(url)
    title = page.find("h1", class_="read__title").text

    ## div format: Kompas.com - 06/05/2021, 13:40 WIB
    datetime_str = page.find('div', class_="read__time").text.split('-')[1].strip()
    date_str = datetime_str.split(',')[0]

    img_content = page.find('div', class_="photo")
    img_link= img_content.find('img').attrs['src']

    contents = page.find('div', class_="read__content")
    paragraphs = contents.find_all('p')

    news_content = {}
    i = 0
    for paragraph in paragraphs:
        paragraph = unicodedata.normalize("NFKD", paragraph.text).strip()

        # Skip empty and 'baca juga' paragraph
        if paragraph != "" and not paragraph.startswith("Baca juga:"):
            news_content[i] = paragraph
            i += 1

    tags = page.find_all('li', class_="tag__article__item")
    tags_list = []
    for tag in tags:
        tags_list.append(tag.text)

    return {
        'portal': 'kompas.com',
        'date': date_str,
        'link': url,
        'title': title,
        'image': img_link,
        'content': news_content,
        'tags': tags_list,
    }

if __name__ == '__main__':
    ## Get list of urls
    date = '06/05/2021'
    urls = get_urls_by_date(date)
    
    ## scrap first url
    news_attr = get_news_attr(urls[0] + '?page=all')
    print(news_attr)



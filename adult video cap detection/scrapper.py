from bs4 import BeautifulSoup as bs
import requests
import json
import time

URL = 'https://www.xnxx.com/tags'

response = requests.get(URL)
print(response.status_code)
titles = []



if response.status_code == 200:
    soup = bs(response.content, 'html.parser')
    tags = soup.select("#tags li a")
    count = 0
    for tag in tags:
      if count == 50:
        break
      newURL = "https://www.xnxx.com{}".format(tag['href'])
      newPage = requests.get(newURL)
      if newPage.status_code == 200:
        newSoup = bs(newPage.content, 'html.parser')
        blockTitles = newSoup.select(".mozaique .thumb-under p a")
        for blockTitle in blockTitles:
          titles.append(blockTitle["title"])
      count += 1
    print(titles)
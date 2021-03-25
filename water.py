from bs4 import BeautifulSoup
import requests
import pandas as pd

base = "https://papers.nips.cc"
html = requests.get("https://papers.nips.cc/paper/2020").text
soup = BeautifulSoup(html, 'html.parser')

data = [*soup.find_all('ul')[1].find_all('li')]
data1 = [*soup.find_all('ul')[1].children]

list_of_papers = []

for item in data1:
    if item != '\n':
        title = item.find('a').text
        url = base + item.find('a').get_attribute_list('href')[0]
        html_abstract = requests.get(url).text
        soup_abstract = BeautifulSoup(html_abstract, 'html.parser')
        paragraphs = soup_abstract.find_all('p')
        data_abstract = [text for text in paragraphs if len(text.text)>20]
        list_of_papers.append({title:data_abstract})

lists = []

for x in list_of_papers:
    for y,z in x.items():
        lists.append([y,z[-1]])


for y,z in list_of_papers[11].items():
        print(y,z[-1])


df = pd.DataFrame(lists, )

# df.drop(index=[*range(10)], axis = 0, inplace  = True)
df.to_csv('/aaaaaaaaaaaaaa.csv')
# df = pd.read_csv("g:/aaaaaaaaaaaaaa.csv ")

soup_abstract1 = BeautifulSoup(lists[12][1][-1], 'html.parser')
paragraphs1 = soup_abstract.find_all('p')
lists[10]
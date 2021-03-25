import nltk
from bs4 import BeautifulSoup
import requests
import pandas as pd
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


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

df.drop(index=[*range(10)], axis = 0, inplace  = True)
df.to_csv('/aaaaaaaaaaaaaa.csv')

#%%
df = pd.read_csv("papers.csv ")
df.drop('Unnamed: 0', axis = 1, inplace = True)
df.columns = ['title','abstract']

#%%
df.abstract = df.abstract.apply(lambda x:re.sub('<.*?>','',x))
df.abstract = df.abstract.apply(lambda x:re.sub(r'(\n|\r)+',' ',x))
df.abstract = df.abstract.apply(lambda x:re.sub(r'  ',' ',x))

#%%

stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")
#%%

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens



#%%
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in df.abstract:
    allwords_stemmed = tokenize_and_stem(i)  # for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed)  # extend the 'totalvocab_stemmed' list

    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

#%%


#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(df.abstract) #fit the vectorizer to synopses

print(tfidf_matrix.shape)


terms = tfidf_vectorizer.get_feature_names()
#%%
from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)

#%%
from sklearn.cluster import KMeans

num_clusters = 5

km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

#%%
from sklearn.externals import joblib

#uncomment the below to save your model
#since I've already run my model I am loading from the pickle

joblib.dump(km,  'doc_cluster.pkl')

# km = joblib.load('doc_cluster.pkl')
# clusters = km.labels_.tolist()
http://brandonrose.org/clustering
look for topic modeling

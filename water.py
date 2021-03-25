import nltk
from bs4 import BeautifulSoup
import requests
import pandas as pd
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
#%%

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
df = pd.read_csv("models/papers.csv ")
df.drop('Unnamed: 0', axis = 1, inplace = True)
df.columns = ['title','abstract']

#%%
df.abstract = df.abstract.apply(lambda x:re.sub('<.*?>','',x))
df.abstract = df.abstract.apply(lambda x:re.sub(r'(\n|\r)+',' ',x))
df.abstract = df.abstract.apply(lambda x:re.sub(r'  ',' ',x))

#%%
df.to_json('papers.json',orient= 'records')
df.to_csv('Nips_papers.csv')
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
                                 use_idf=True, tokenizer=tokenize_only, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(df.abstract) #fit the vectorizer to synopses

print(tfidf_matrix.shape)


terms = tfidf_vectorizer.get_feature_names()
#%%
from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)

#%%
from sklearn.cluster import KMeans

num_clusters = 50

km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

#%%
from sklearn.externals import joblib

#uncomment the below to save your model
#since I've already run my model I am loading from the pickle

joblib.dump(km,  'cluster50_token_only.pkl')

km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()


#%%
import json
import os
from sentence_transformers import SentenceTransformer, util

#First, we load the papers dataset (with title and abstract information)
dataset_file = 'models/papers.json'

# if not os.path.exists(dataset_file):
  # util.http_get("https://sbert.net/datasets/emnlp2016-2018.json", dataset_file)

with open(dataset_file) as fIn:
  papers = json.load(fIn)

print(len(papers), "papers loaded")

#%%
#We then load the allenai-specter model with SentenceTransformers
model = SentenceTransformer('allenai-specter')
#To encode the papers, we must combine the title and the abstracts to a single string
paper_texts = [paper['title'] + ' ' + paper['abstract'] for paper in papers]
model.loa
#Compute embeddings for all papers
corpus_embeddings = model.encode(paper_texts, convert_to_tensor=True)
#%%
# We define a function, given title & abstract, searches our corpus for relevant (similar) papers
def search_papers(title, abstract):
    query_embedding = model.encode(title + ' ' + abstract, convert_to_tensor=True)

    search_hits = util.semantic_search(query_embedding, corpus_embeddings)
    search_hits = search_hits[0]  # Get the hits for the first query

    print("Paper:", title)
    print("Most similar papers:")
    for hit in search_hits:
        related_paper = papers[hit['corpus_id']]
        print("{:.2f}\t{}".format(hit['score'], related_paper['title'], ))

#%%

# This paper was the EMNLP 2019 Best Paper
search_papers(title='Self-Supervised MultiModal Versatile Networks',
              abstract='Videos are a rich source of multi-modal supervision. In this work, we learn representations using self-supervision by leveraging three modalities naturally present in videos: visual, audio and language streams. To this end, we introduce the notion of a multimodal versatile network -- a network that can ingest multiple modalities and whose representations enable downstream tasks in multiple modalities. In particular, we explore how best to combine the modalities, such that fine-grained representations of the visual and audio modalities can be maintained, whilst also integrating text into a common embedding. Driven by versatility, we also introduce a novel process of deflation, so that the networks can be effortlessly applied to the visual data in the form of video or a static image. We demonstrate how such networks trained on large collections of unlabelled video data can be applied on video, video-text, image and audio tasks. Equipped with these representations, we obtain state-of-the-art performance on multiple challenging benchmarks including UCF101, HMDB51, Kinetics600, AudioSet and ESC-50 when compared to previous self-supervised work. Our models are publicly available.')

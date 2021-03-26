import nltk
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import re
import json
import os
from sentence_transformers import SentenceTransformer, util
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
#%%

#First, we load the papers dataset (with title and abstract information)
dataset_file = 'models/papers.json'

# if not os.path.exists(dataset_file):
  # util.http_get("https://sbert.net/datasets/emnlp2016-2018.json", dataset_file)

with open(dataset_file) as fIn:
  papers = json.load(fIn)

print(len(papers), "papers loaded")
#%%
#We then load the allenai-specter model with SentenceTransformers
# model = SentenceTransformer('allenai-specter')
#To encode the papers, we must combine the title and the abstracts to a single string
paper_texts = [paper['title'] + ' ' + paper['abstract'] for paper in papers]

#Compute embeddings for all papers
corpus_embeddings = model.encode(paper_texts, convert_to_tensor=True)
#%%

corpus_embeddings = np.load('/content/drive/MyDrive/Ms/trans/embed.npy')
model = SentenceTransformer("/content/drive/MyDrive/Ms/trans")

# We define a function, given title & abstract, searches our corpus for relevant (similar) papers
def search_papers(title, abstract):
    query_embedding = model.encode(title + ' ' + abstract, convert_to_tensor=True)

    search_hits = util.semantic_search(query_embedding, corpus_embeddings)
    search_hits = search_hits[0]  # Get the hits for the first query
util.semantic_search(5,6,)
    print("Paper:", title)
    print("Most similar papers:")
    for hit in search_hits:
        related_paper = papers[hit['corpus_id']]
        print("{:.2f}\t{}".format(hit['score'], related_paper['title'], ))

#%%

# This paper was the EMNLP 2019 Best Paper
search_papers(title='Self-Supervised MultiModal Versatile Networks',
              abstract='Videos are a rich source of multi-modal supervision. In this work, we learn representations using self-supervision by leveraging three modalities naturally present in videos: visual, audio and language streams. To this end, we introduce the notion of a multimodal versatile network -- a network that can ingest multiple modalities and whose representations enable downstream tasks in multiple modalities. In particular, we explore how best to combine the modalities, such that fine-grained representations of the visual and audio modalities can be maintained, whilst also integrating text into a common embedding. Driven by versatility, we also introduce a novel process of deflation, so that the networks can be effortlessly applied to the visual data in the form of video or a static image. We demonstrate how such networks trained on large collections of unlabelled video data can be applied on video, video-text, image and audio tasks. Equipped with these representations, we obtain state-of-the-art performance on multiple challenging benchmarks including UCF101, HMDB51, Kinetics600, AudioSet and ESC-50 when compared to previous self-supervised work. Our models are publicly available.')

#%%

import urllib
urllib.urlretrieve ("https://drive.google.com/drive/folders/1keMoWi6k4ixxCgxZTt9Fqe-1B1nw6oAi?usp=sharing", "mp3.mp3")

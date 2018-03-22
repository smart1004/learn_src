import codecs
# from bs4 import BeautifulSoup
# from konlpy.tag import Twitter
from gensim.models import word2vec

model = word2vec.Word2Vec.load("toji.model")
model.most_similar(positive=["땅"])

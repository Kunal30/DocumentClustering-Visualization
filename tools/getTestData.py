from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import reuters
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
labels = reuters.categories()

n_classes = 90


from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import nltk
import pandas as pd
from sklearn.manifold import TSNE
import time
# from tsne import tsne
import pickle
from preprocess_reuters_dataset import get_medium_dataset
from gensim.models import LdaMulticore
# '''

def main():

	lda_model=LdaMulticore.load('lda.model')
	print('Successfully Loaded')
	print(lda_model)

	f = open('cnn_text.pickle', 'r')
	test_data1 = pickle.load(f)
	
	f = open('test_dataset.txt', 'r')
	test_data2 = pickle.load(f)
	
	

	



if __name__ == "__main__":
	main()		
# Code Author: Kunal Vinay Kumar Suthar
# ASU ID: 1215112535
# Course: CSE-573: Semantic Web Mining
# Project: Document Clustering and Visualization

# 1) ----> Data preprocessing(Tokenization, Stemming, Stopword Removal, Lematization) 
# 2) ----> Latent Dirichlet Allocation 
# 3) ----> TSNE 
# 4) ----> 2D Visualization 
# 5) ----> 3D Visualization


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
from preprocess_reuters_dataset import get_medium_dataset
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def main():
	feature_vectors=np.load("input2tsne.npy")
	topics=np.zeros((feature_vectors.shape[0],1))

	print(feature_vectors.shape)
	class_count_Z=np.zeros((20,1))
	for i in range(feature_vectors.shape[0]):
		# print(feature_vectors[i])
		# print(np.argmax(feature_vectors[i]))
		topics[i]=np.argmax(feature_vectors[i])
		# print(topics[i])
		class_count_Z[int(topics[i])]=class_count_Z[int(topics[i])]+1
	# print(topics)
	# print(topics.shape)	
	# print(class_count_Z)
	color=['blue','green','red','cyan',
	'magenta','yellow','black','#293f63',
	'#2a8c25','#c6c431','#aa317e','#68271c',
	'#f27900','#0be09c','#ba9e9e','#a31f01',
	'#42b765','#1c98a8','#32e0ff','#a5568f']

	latent=TSNE(n_components=2).fit_transform(feature_vectors)
	print(latent.shape)
	
	f = open("latent.txt", "w")
	pickle.dump(latent,f)
	f.close()

	f = open("Z.txt", "w")
	pickle.dump(class_count_Z,f)
	f.close()

	f = open("topics.txt", "w")
	pickle.dump(topics,f)
	f.close()

	for i in range(20):
		ix=np.where(topics==i)
		plt.scatter(latent[ix,0],latent[ix,1],c=color[i])
	
	plt.title("LDA on 20 Newsgroups dataset")
	plt.xlabel("Latent Semantics x[0]")
	plt.ylabel("Latent Semantics x[1]")
	plt.show()
	





if __name__ == "__main__":
	main()		
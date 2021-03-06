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
from tsne import tsne
from preprocess_reuters_dataset import get_medium_dataset

'''
Write a function to perform the pre processing steps on the entire dataset
'''
def lemmatize_stemming(text):
    stemmer= SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result


def load_reuters_data():
    """
    Load the Reuters dataset.

    Returns
    -------
    Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    nltk.download('stopwords')
    nltk.download('reuters')
    stop_words = stopwords.words("english")
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    mlb = MultiLabelBinarizer()
    
    documents = reuters.fileids()
    test = [d for d in documents if d.startswith('test/')]
    train = [d for d in documents if d.startswith('training/')]

    docs = {}
    docs['train'] = [reuters.raw(doc_id) for doc_id in train]
    docs['test'] = [reuters.raw(doc_id) for doc_id in test]
    
    return docs


def main():
	
	# Fetching newsgroups20 train and testing data using sklearn's API

	newsgroups_train = fetch_20newsgroups(subset='train')
	newsgroups_test = fetch_20newsgroups(subset='test')
	
	
	# Fetching newsgroups20 train and testing data using nltk's API
	data=load_reuters_data()
	reuters_train= data['train']
	reuters_test= data['test']


	medium_total=get_medium_dataset()

	med_len=(medium_total.size)/2
	medium_train=medium_total[0:med_len]
	medium_test=medium_total[med_len:-1]

	# print(reuters_train[0])
	print('*********************************')
	nltk.download('wordnet')
	

	#Accumulating data from both the datasets
	total_train=[]
	total_test=[]

	# Developing the train set
	for doc in newsgroups_train.data:
		total_train.append(doc)

	for doc in reuters_train:
		total_train.append(doc)
	
	for doc in medium_train:
		total_train.append(doc)


	# Developing the test set	
	for doc in newsgroups_test.data:
		total_test.append(doc)

	for doc in reuters_test:
		total_test.append(doc)

	for doc in medium_test:
		total_test.append(doc)

		
	# total_train.append(newsgroups_train.data)		
	# total_train.append(reuters_train)		
	# total_test.append(newsgroups_test.data)		
	# total_test.append(reuters_test)		
	# print(newsgroups_train.data[1])
	print('Data has been totaled!!')
	# print(newsgroups_test)

	#Preprocessing the training data
	processed_docs = []
	
	# for doc in newsgroups_train.data:
	# 	processed_docs.append(preprocess(doc))

	# print(total_train)
	print('#######################################################')
	for doc in total_train:
		# print(doc)
		processed_docs.append(preprocess(doc))

	'''
	Create a dictionary from 'processed_docs' containing the number of times a word appears 
	in the training set using gensim.corpora.Dictionary and call it 'dictionary'
	'''
	dictionary = gensim.corpora.Dictionary(processed_docs)
	
	# '''
	# Checking dictionary created
	# '''
	# count = 0
	# for k, v in dictionary.iteritems():
	#     print(k, v)
	#     count += 1
	#     if count > 10:
	#         break

	'''
	Create the Bag-of-words model for each document i.e for each document we create a dictionary reporting how many
	words and how many times those words appear. Save this to 'bow_corpus'
	'''
	bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
	
	'''
	Preview BOW for our sample preprocessed document
	'''
	# document_num = 20
	# bow_doc_x = bow_corpus[document_num]

	# for i in range(len(bow_doc_x)):
	#     print("Word {} (\"{}\") appears {} time.".format(bow_doc_x[i][0], 
	#                                                      dictionary[bow_doc_x[i][0]], 
	#                                                      bow_doc_x[i][1]))        
	
	'''
	Train your lda model using gensim.models.LdaMulticore and save it to 'lda_model'
	'''
	# TODO
	lda_model =  gensim.models.LdaMulticore(bow_corpus, 
	                                   num_topics = 20, 
	                                   id2word = dictionary,                                    
	                                   passes = 10,
	                                   workers = 2)


	'''
	For each topic, we will explore the words occuring in that topic and its relative weight
	'''
	for idx, topic in lda_model.print_topics(-1):
	    print("Topic: {} \nWords: {}".format(idx, topic ))
	    print("\n")

	# num = 100
	# unseen_documents=[]
	# for i in range(0,len(newsgroups_test.data)):
	# 	unseen_documents.append(newsgroups_test.data[i])
	
	# print(unseen_documents)    

	# Data preprocessing step for the unseen document
	for doc in total_train:
		bow_vector = dictionary.doc2bow(preprocess(doc))
		print("*********************************************************************")
		print(doc)
		print("*********************************************************************")
		for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
		    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))

	
	#Saving the trained LDA model
	lda_model.save('lda.model')
	


if __name__ == "__main__":
	main()	
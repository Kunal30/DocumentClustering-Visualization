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
from gensim.models import LdaMulticore
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

def main():

	lda_model=LdaMulticore.load('lda.model')
	print('Successfully Loaded')
	print(lda_model)

	feature_vectors=np.array([])
	#i=0
	#for doc in newsgroups_test.data:
	#	#i=i+1
	#	bow_vector = dictionary.doc2bow(preprocess(doc))
	#	#print("*********************************************************************")
	#	#print(doc)
		
	#	#print(lda_model[bow_vector])
	#	#print("*********************************************************************")          
	#	distribution=lda_model.get_document_topics(bow_vector, minimum_probability=0.0, minimum_phi_value=None, per_word_topics=False)
	#	onerow=[]
	#	for j in range(20):
	#		onerow.append(distribution[j][1])
	#	onerow=np.array(onerow)f

	#	#print(type(feature_vectors))
	#	if(feature_vectors.size==0):
	#		feature_vectors=onerow
	#	else:
	#		feature_vectors=np.vstack((feature_vectors,onerow))

	#	del onerow
	#	#for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
	#	#	#print(index)
	#	#	#print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index)))
 # #          list_of_prob=lda_model.show_topic(index, len(bow_vector))
 # #          k
	#	#	print(len(bow_vector))
	#	#	print(len(lda_model.show_topic(index, len(bow_vector))))
	#	#	print(lda_model.show_topic(index, len(bow_vector)))
	#	#if i>=3:
	#	#	break

	##print(feature_vectors)
	#print(feature_vectors.shape)
	#np.save("input2tsne",feature_vectors)

if __name__ == "__main__":
	main()		
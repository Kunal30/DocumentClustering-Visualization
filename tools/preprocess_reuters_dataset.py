# Code Author: Kunal Vinay Kumar Suthar
# ASU ID: 1215112535
# Course: CSE-573: Semantic Web Mining
# Project: Document Clustering and Visualization
# In this module we are preprocessing the dataset from medium comprising of machine Learning articles
import numpy as np
import pandas as pd
import random
from random import shuffle
import math
from collections import Counter
import matplotlib.pyplot as plt
import operator

def get_medium_dataset():
	# print('hello')
	data=pd.read_csv('articles.csv')
	# print(data)
	# print(data['text'][0])

	return data['text']


















if __name__=="__main__":
	main()	
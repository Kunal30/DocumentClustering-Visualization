from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
color=['blue','green','red','cyan',
'magenta','yellow','black','#293f63',
'#2a8c25','#c6c431','#aa317e','#68271c',
'#f27900','#0be09c','#ba9e9e','#a31f01',
'#42b765','#1c98a8','#32e0ff','#a5568f']
# df=np.load('input2tsne.npy')
# print(df.shape)
# datasets=fetch_20newsgroups(subset='test')
# label=datasets.target
# print(label[0:10])
# print(color[label[0]],color[label[1]],color[label[2]])
# latent=TSNE(n_components=2).fit_transform(df)
# # for i in range(100):
# 	# for j in range(20): 
# 	# plt.scatter(latent[i,0],latent[i,1],c=color[label[i]])
		
# # plt.show()
# # label=label[0:100]
# for i in range(20):
# 	ix=np.where(label==i)
# 	plt.scatter(latent[ix,0],latent[ix,1],c=color[i])
# plt.show()

dataset = fetch_20newsgroups(subset='train',shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data
print(len(documents))
no_features = 1000

# NMF is able to use tf-idf
# tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
# tfidf = tfidf_vectorizer.fit_transform(documents)
# tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0)
pred=lda.fit_transform(tf)
latent = TSNE(n_components=2).fit_transform(pred)
label=dataset.target
for i in range(20):
	ix=np.where(label==i)
	plt.scatter(latent[ix,0],latent[ix,1],c=color[i])
plt.title("LDA on 20 Newsgroups dataset")
plt.xlabel("Latent Semantics x[0]")
plt.ylabel("Latent Semantics x[1]")
plt.show()
np.save("input2tsne_train",pred)
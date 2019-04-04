import time
from sklearn.manifold import TSNE


def tsne(data):
	
	n_sne = 7000
	time_start = time.time()
	tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
	tsne_results = tsne.fit_transform(data)
	print(tsne.shape)
	print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)
from sklearn.manifold import TSNE
import numpy as np

class tsne_object():
    def __init__(self, dataframe, feature_vec, n_components=2, init='pca', random_state=1, method='exact', n_iter=200, verbose=20):
        '''INPUT: pandas dataframe, 2D numpy feature vec of embedding space trained with gensim word2vec or doc2vec, parameters for sklearn TSNE model.'''
        self.feature_vec = feature_vec
        #Load TSNE with input parameters
        tsne = TSNE(n_components=2, init='pca', random_state=1, method='exact', n_iter=200, verbose=20)
        #fit and transform feature vec to 2d TSNE vec
        self.tsne_vec = tsne.fit_transform(self.feature_vec)

    def plot_tsne(self, title=None):
        '''INPUT: Optional Title
           Output: Plot of TSNE embedding colored by thread membership.'''
        x_min, x_max = np.min(self.tsne_vec, 0), np.max(self.tsne_vec, 0)
        X = (self.tsne_vec - x_min) / (x_max - x_min)

        plt.figure(figsize=(15,15))
        ax = plt.subplot(111)
        for i in range(X.shape[0]):
            plt.text(X[i, 0], X[i, 1], str(features.thread_ids[i]),
                     color=plt.cm.Set1(y[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 20})

        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)


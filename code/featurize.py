import numpy as np
import pandas as pd
import preprocess
import spacy
from collections import Counter
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import preprocess
from nltk.tokenize import sent_tokenize
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

from sklearn.metrics import confusion_matrix
from gensim_modeling import *

class Corpus_Features(object):
    def __init__(self, dataframe, post_counts, tag_counts):
        '''Input: row from a pandas dataframe, bag of words counter, and tag counter'''
        self.df = dataframe
        #Join tokens into strings
        self.posts = self.df.tokens.apply(lambda x: ' '.join(x)).values
        self.tags = self.df.tags
        self.post_counts = post_counts
        self.tag_counts = tag_counts
        self.threads = self.df.thread

        #Encode thread labels to numbers for classification
        self.thread_dict = {'biology':0,'cooking':1,'crypto':2,'diy':3,'robotics':4,'travel':5}
        self.thread_ids = self.df.thread.apply(lambda x: self.thread_dict[x]).values

    def bag_of_words(self):
        """Count vectorize every question in the dataset."""
        cv = CountVectorizer()
        self.bow  = cv.fit_transform(self.posts)
        

    def tfidf(self):
        """Create tfidf vectors for every question in the dataset"""
        tfidf = TfidfVectorizer()
        self.tfidf_vec = tfidf.fit_transform(self.posts)

    def word2vec(self, feature, sample=1e-3, size=128,window=10, min_count=0, workers=4):
        """train a word2vec model on a series of tokens. Feature label must be a string."""
        sentences = list(self.df[feature])
        model = Word2Vec(sentences, workers=workers, size=size, min_count=min_count, window=window,sample=sample)

        return model

    def d2v(self):
        '''Train doc2vec model on dataframe. Uses modules that can be found in gensim_modeling.py'''
        model = Doc2Vec(self.dataframe)
        return model


    def most_frequent_tags(self,n_tags):
        """Returns the most frequent tags in the class object."""

        top_n_tags = sorted(self.tag_counts, key=self.tag_counts.get, reverse=True)[:n_tags]

        return top_n_tags

    def most_frequent_tokens(self, n_tokens):
        """Returns the most frequent tokens in the class object."""

        top_n_tokens = sorted(self.post_counts, key=self.post_counts.get, reverse=True)[:n_tokens]

        return top_n_tokens      

    def tags_in_post(self):
        '''Creates Bag of Words feature with a 1 or 0 if tag is in the title of each document'''
        #place to store matrix
        tag_title_matrix = np.zeros(shape=(len(self.df), len(self.tag_counts)))
        #Populate feature matrix with binary labels
        for j,tag in enumerate(self.tag_counts.keys()):
            for i,post in enumerate(self.df.tokens.values):
                if tag in post:
                  tag_title_matrix[i,j] = 1

        return tag_title_matrix

    def tag_freq_in_post(self):
        '''Creates term-frequency feature with counts of the number of times a tag appears in a post'''
        #place to store matrix
        tag_content_matrix = np.zeros(shape=(len(self.df), len(self.tag_counts)))
        #Populate feature matrix with tag counts
        for j,tag in enumerate(self.tag_counts.keys()):
            for i,post in enumerate(self.df.tokens.values):
                token_counter = Counter(post)
                if tag in post:
                    tag_content_matrix[i,j] = token_counter[tag]

        return tag_content_matrix

def makeFeatureVec(words, model, num_features):
    """Average all the word2vec vectors in a document"""
    
    # Place to store matrix
    featureVec = np.zeros((num_features,),dtype="float32")
    
    #Store word count for subsequent averaging
    nwords = 0
     
    #index2word contains the words in the model. Taking its set reduces redundancy
    index2word_set = set(model.index2word)
    
    # Loop over each word in the review and, if it is in the model's
    # vocabulary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
     
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(docs, model, num_features):
    """Calculate average word2vec vectors for each document and return an embedding matrix"""

    # Preallocate a 2D numpy array, for speed
    docFeatureVecs = np.zeros((len(docs),num_features),dtype="float32")
     
    # Get averaged embedding for each doc
    for d in docs:
        docFeatureVecs[counter] = makeFeatureVec(d, model, \
           num_features)
    
    return docFeatureVecs

if __name__ == '__main__':
    data_file = open("pickle_jar/dataframes/allthreads_first1000_data.p", 'rb')
    data = pickle.load(data_file)

    tag_token_counter_file = open('pickle_jar/tag_counters/allthreads_first1000_minimal_processing_tag_dict.p', 'rb')
    tag_token_counter = pickle.load(tag_token_counter_file)

    post_token_counter_file = open('pickle_jar/post_counters/allthreads_first1000_minimal_processing_ctxt_dict.p', 'rb')
    post_token_counter = pickle.load(post_token_counter_file)

    features = Corpus_Features(data, post_token_counter, tag_token_counter)

    print("# of docs: {}".format(len(features.df)))
    print("# of unique tags: {}".format(len(features.tag_counts)))
    #print("total word count: {}".format(features.df.tokens))
    print("# of unique words: {}".format(len(features.post_counts)))


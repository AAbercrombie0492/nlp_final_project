import numpy as np
import pandas as pd
import sklearn
import spacy
from spacy import attrs
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import time
from collections import Counter, defaultdict
import _pickle as cPickle

from preprocess import *
from featurize import *
from model import *
from gensim_modeling import *
from tsne_viz import *
#import model

class Pipeline(object):
    """Machine learning pipeline that serves as a platform to load, process, featurize, and model stack exchange data. Modeling techniques aim to classify and autotag documents."""

    def __init__(self, name):
        self.name = name

    def preprocess(self,data_directory='data', files=['biology.csv'], n_samples_per_file=100, rem_stopwords=True, triple_title=True, ner=True, rem_punc = True,lemmatize=True):
        """Data cleaning pipeline: Word segmentation stemming, correct mispellings. Apply unicode sandwhich. Tokenize data. Remove stop words. Lemmatize. Named Entity Recognition. Filter POS tags."""

        #Load, apply, and store preprocessing objectives from preprocess.py
        preprocess_data = ProcessedData(data_directory, files, n_samples_per_file, rem_stopwords, triple_title, ner, rem_punc,lemmatize)

        #Save to pickle
        self.dataframe = preprocess_data.data
        preprocess_data.data.to_pickle('pickle_jar/dataframes/{}_dataframe.p'.format(self.name))

        #Store token Counter object and pickle it.
        self.token_counts = preprocess_data.word_counts
        with open(r"pickle_jar/post_counters/{}_token_counts.p".format(self.name), "wb") as output_file:
            cPickle.dump(preprocess_data.word_counts, output_file)

        #Store tag Counter object and pickle it.
        self.tag_counts = preprocess_data.tag_counts
        with open(r"pickle_jar/tag_counters/{}_tag_counts.p".format(self.name), "wb") as output_file:
            cPickle.dump(preprocess_data.tag_counts, output_file)

    def featurize(self, vectorizer='tfidf', predict_target='threads_bool'):
        """Platform for vectorizing text data. Options include Bag of words model (bow). TFIDF vectors (tfidf). Word2Vec (word2vec), and Doc2Vec (doc2vec)."""

        #instatiate feature object from featurize.py
        features = Corpus_Features(self.dataframe, self.token_counts, self.tag_counts)
        
        if vectorizer == 'tfidf':
            self.X = features.tfidf()
        elif vectorizer == 'bow':
            self.X = features.bag_of_words()
        elif vectorizer == 'word2vec':
            self.X = features.Word2Vec('ttl_ctxt')
        elif vectorizer == 'doc2vec':

        if predict_target == 'thread':
            self.y = features.thread_ids()
        elif predict_target == 'tags_counts':
            self.y = features.tag_freq_in_post()
        elif predict_target == 'tags_bool':
            self.y = features.tags_in_post()


    def model(self, model, gridsearch_params, scoring='neg_mean_squared_error'):
        """Instantiates the Classify_Thread module from model.py, which splits our data into train and test splits. Enables multiclass classification with Multinomial Naive Bayes, Random Forests, Gradient Boosted Trees, ElasticNet One vs All Logistic Regression, and linear One vs All Support Vector Classifiers.
        """

        self.Classifier = Classify_Thread(self.X, self.y)

        self.model = self.Classifier.model()


    def evaluate(self, k):
        """Performs Kfold cross-validation on model and returns performance metrics and a confusion matrix. Uses modules from model.py"""

        #Peforks kfold cross-val and returns a dict of average performance on train and validation splits. Metrics include accuracy, precision, recall, and f1

        cross_val_dict = kfold_evaluation(X,y,model, kf)

        #load into pandas df
        cross_val_dataframe = pd.DataFrame.from_dict(cross_val_dict)

        return cross_val_dataframe

if __name__ == '__main__':
    print("Processing biology....")
    bio = Pipeline(name='bio_10000_dl')
    bio.preprocess(data_directory='data', files=['biology.csv'], n_samples_per_file=10000, rem_stopwords=True, triple_title=False, ner=False, rem_punc = False,lemmatize=False)

    print("Processing travel...")
    travel = Pipeline(name='travel_10000_dl')
    travel.preprocess(data_directory='data', files=['travel.csv'], n_samples_per_file=10000, rem_stopwords=True, triple_title=False, ner=False, rem_punc = False,lemmatize=False)

    print("classify 1 ...")
    classify_noStopwords_3xTitle_ner_stripPunct = Pipeline(name='classify_noStopwords_3xTitle_ner_stripPunct')
    classify_noStopwords_3xTitle_ner_stripPunct.preprocess(data_directory='data', files=['biology.csv','cooking.csv','crypto.csv','diy.csv','robotics.csv','travel.csv'], n_samples_per_file=1000, rem_stopwords=True, triple_title=False, ner=True, rem_punc = False,lemmatize=False)

    print("classify 2 ...")
    classify_Stopwords_3xTitle_ner_stripPunct = Pipeline(name='classify_Stopwords_3xTitle_ner_stripPunct')
    classify_Stopwords_3xTitle_ner_stripPunct.preprocess(data_directory='data', files=['biology.csv','cooking.csv','crypto.csv','diy.csv','robotics.csv','travel.csv'], n_samples_per_file=1000, rem_stopwords=False, triple_title=False, ner=True, rem_punc = False,lemmatize=False)

    #------- No NER
    print("classify 1 ...")
    classify_noStopwords_3xTitle_Noner_stripPunct = Pipeline(name='classify_noStopwords_3xTitle_Noner_stripPunct')
    classify_noStopwords_3xTitle_Noner_stripPunct.preprocess(data_directory='data', files=['biology.csv','cooking.csv','crypto.csv','diy.csv','robotics.csv','travel.csv'], n_samples_per_file=1000, rem_stopwords=True, triple_title=False, ner=False, rem_punc = False,lemmatize=False)

    print("classify 2 ...")
    classify_Stopwords_3xTitle_Noner_stripPunct = Pipeline(name='classify_Stopwords_3xTitle_Noner_stripPunct')
    classify_Stopwords_3xTitle_Noner_stripPunct.preprocess(data_directory='data', files=['biology.csv','cooking.csv','crypto.csv','diy.csv','robotics.csv','travel.csv'], n_samples_per_file=1000, rem_stopwords=False, triple_title=False, ner=False, rem_punc = False,lemmatize=False)



    #---- LEMMATIZE
    print("classify 3 ...")
    classify_noStopwords_3xTitle_ner_stripPunct_lemmatize = Pipeline(name='classify_noStopwords_3xTitle_ner_stripPunct_lemmatize')
    classify_noStopwords_3xTitle_ner_stripPunct_lemmatize.preprocess(data_directory='data', files=['biology.csv','cooking.csv','crypto.csv','diy.csv','robotics.csv','travel.csv'], n_samples_per_file=1000, rem_stopwords=True, triple_title=False, ner=True, rem_punc = False,lemmatize=True)

    print("classify 4 ...")
    classify_Stopwords_3xTitle_ner_stripPunct_lemmatize = Pipeline(name='classify_Stopwords_3xTitle_ner_stripPunct_lemmatize')
    classify_Stopwords_3xTitle_ner_stripPunct_lemmatize.preprocess(data_directory='data', files=['biology.csv','cooking.csv','crypto.csv','diy.csv','robotics.csv','travel.csv'], n_samples_per_file=1000, rem_stopwords=False, triple_title=False, ner=True, rem_punc = False,lemmatize=True)

    #------- No NER
    print("classify 5 ...")
    classify_noStopwords_3xTitle_Noner_stripPunct_lemmatize = Pipeline(name='classify_noStopwords_3xTitle_Noner_stripPunct_lemmatize')
    classify_noStopwords_3xTitle_Noner_stripPunct_lemmatize.preprocess(data_directory='data', files=['biology.csv','cooking.csv','crypto.csv','diy.csv','robotics.csv','travel.csv'], n_samples_per_file=1000, rem_stopwords=True, triple_title=False, ner=False, rem_punc = False,lemmatize=True)

    print("classify 6 ...")
    classify_Stopwords_3xTitle_ner_stripPunct_lemmatize = Pipeline(name='classify_Stopwords_3xTitle_ner_stripPunct_lemmatize')
    classify_Stopwords_3xTitle_ner_stripPunct_lemmatize.preprocess(data_directory='data', files=['biology.csv','cooking.csv','crypto.csv','diy.csv','robotics.csv','travel.csv'], n_samples_per_file=1000, rem_stopwords=False, triple_title=False, ner=False, rem_punc = False,lemmatize=True)








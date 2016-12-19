'''Must be run in a python 2 environment'''

from __future__ import division

import graphlab as gl
import pandas as pd
import pyLDAvis
import pyLDAvis.graphlab
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import re

import numpy as np
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
import pickle
import ftfy
import chardet
import matplotlib as plt

#nlp = spacy.en.English()
pyLDAvis.enable_notebook()


class lda_object():
    def __init__(self, dataframe):
        '''Must be run in a python 2 environment.
        INPUT: Cleaned and preprocessed pandas dataframe'''
        self.dataframe = dataframe
        #Graphlab LDA needs a bag of words dictionary for each document in the dataset.
        self.dataframe['bow'] = dataframe.ttl_ctxt.apply(lambda x: dict(Counter(x.split())))
        #Graphlab also require a sframe object
        self.sframe = gl.load_sframe(dataframe)
        self.bow = self.sframe['bow']

        def topic_modelling(self,n_topics, n_iterations):
            #Train Graphlab topic model
            topic_model = gl.topic_model.create(self.bows, num_topics, num_iterations)
            return topic_model

        def lda_vis(self,topic_model):
            #Visualize graphlab topic model
            plt.figure()
            pyLDAvis.graphlab.prepare(topic_model, self.bows)
            plt.show()

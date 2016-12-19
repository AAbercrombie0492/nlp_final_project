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
import chardet
import ftfy

nlp = spacy.en.English()

class ProcessedData():

    def __init__(self, data_directory='data', files=['biology.csv'], n_samples_per_file=100, rem_stopwords=True, triple_title=True, ner=True, rem_punc = True,lemmatize=True):
        """This class serves as a platform to store and process stack exchange data. Required input is a filepath and a list of desired csvs to process."""

        print("\nCommencing data processing.....")
        self.dir = data_directory
        self.files = files
        self.data = pd.DataFrame([])

        print("\nReading in files....")
        start = time.clock()
        self.read_files(n_samples_per_file)
        print("Time to read in files:\t\t",time.clock()-start)

        print("\nConcatenating title + context...")
        start = time.clock()
        if triple_title == True:
            self.triple_title()
            self.concat_features(['triple_title','content'])
        else:
            self.concat_features(['title','content'])
        print("Time to concat title + context:\t\t",time.clock()-start)

        print("\nProcessing data...")
        start = time.clock()
        self.preprocess(rem_stopwords, triple_title, ner, rem_punc,lemmatize)
        print("Time to process data:\t\t",time.clock()-start)

    def read_files(self,n_samples_each_file):
        """Read in csv files, convert to pandas dataframe, and concatenate to master dataframe"""
        
        for f in self.files:
            #Get the name of the thread for each document
            thread = f.split('.')[0]
            df = pd.read_csv(self.dir + '/{}'.format(f))
            #Assign all documents a thread label
            df['thread']=thread
            #Add to master datafarme
            self.data = pd.concat([self.data, df.iloc[:n_samples_each_file]])

    def triple_title(self):
        """Takes a string of title text and duplicates it 3 times."""
        self.data['triple_title'] = (' '+self.data['title']) *3

    def concat_features(self, features):
        """Concatenates two columns from a pandas dataframe and assigns to ttl_ctxt feature."""
      # '''INPUT: list of features to be joined.
      # OUTPUT: New column in data of joined features.'''

        baseline = self.data[features[0]]
        concatenated = baseline.str.cat(''+self.data[features[1]])
        self.data['ttl_ctxt'] = concatenated

    def preprocess(self,rem_stopwords=True, triple_title=True, ner=True, rem_punc=True,lemmatize=True):
        """Apply preprocessing functions to dataframe"""

        #Tokenize text
        self.data['tokens'] = self.data.loc[:,'ttl_ctxt'].apply(lambda x: clean_data(x, rem_stopwords=rem_stopwords, ner=ner, rem_punc=rem_punc, lemmatize=lemmatize))

        #Clean up tags
        self.data.loc[:,'tags'] = self.data.loc[:,'tags'].apply(lambda x: clean_text(x))

        #Store word_count and tag_count dictionaries
        self.word_counts = Counter(np.hstack(self.data.tokens.values))
        self.tag_counts = Counter(np.hstack(self.data.tags.values))

def remove_stopwords(text):
    """Removes stopwords from text as define by nltk"""
    tokens = text.split()
    output = []
    for t in tokens:
        if t not in stopwords.words():
          output.append(t)
    return ' '.join(output)

def remove_punctuation(text):
    """Remove hairy punctuation from text, returning clean text"""
    #Remove punctuation
    text_clean = re.sub(r'[^\w\s]',' ',text)
    #Remove "\n"
    text_clean = re.sub(r'(\n)', '',text_clean)
    #Remove "\t"
    text_clean = re.sub(r'(\\t)', '',text_clean)
    #Remove "\x"
    text_clean = re.sub(r'(\\x)', '',text_clean)
    return text_clean


def capitalize_named_entities(text):
    """Identify named entities in text and convert word to all-caps if the word is a named entity"""
    #spacy_parser = spacy.en.English()
    spacy_text = nlp(text)
    output = []
    for t in spacy_text:
        if t.ent_type_ not in ['', 'DATE']:
            output.append(t.text.upper())
        else:
          output.append(t.text.lower())
    return ' '.join(output)


def lemmatize_text(text):
    """Lemmatize text"""
    wordnet_lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    output = []
    #Lemmatize each token and store in a bucket
    for t in tokens:
        lemmanade =  wordnet_lemmatizer.lemmatize(t)
        lemmanade = wordnet_lemmatizer.lemmatize(lemmanade, pos='v')
        output.append(lemmanade)
    
    #Join lemmatized tokens back into string
    return ' '.join(output)

def clean_text(x):
    '''General purpose text cleaner that reads in a string, applies Unicode sandwhich, strips special characters, and returns a list of tokens.'''

    #Fix encoding in case there is a problem
    x = ftfy.fix_encoding(x)
    #Unicode Sandwhich...
    x = x.encode('UTF-8') #encode as UTF-8
    input_code = chardet.detect(bytes(x))['encoding'] #Auto-detect encoding
    u_string = x.decode(input_code) #Decode
    re.sub(r'\w{4}', u'xxxx', u_string, flags=re.UNICODE) #Handle funny unicode artifacts
    x = u_string.encode('UTF-8') #Encode back to UTF-8
    x = str(x) #Convert to string
    
    x = re.sub(r'[\[\]\"\'\,]',' ', x) #Remove backslashes
    x = re.sub(r'[^\w\s]',' ',x) #Insert space after end of each sentence
    x = x.split() #split string to list
    return x


def clean_data(x, rem_stopwords=True, ner=True, rem_punc=True, lemmatize=False):
    '''Text cleaner that applies preprocessing rules to text. Preprocessing options include, ner, stop word removal, lemmatization, html cleaning, and punctuation removal.'''

    #Unicode Sandwhich
    #Fix encoding in case there is a problem
    x = ftfy.fix_encoding(x)
    #Unicode Sandwhich...
    x = x.encode('UTF-8') #encode as UTF-8
    input_code = chardet.detect(bytes(x))['encoding'] #Auto-detect encoding
    try:
        u_string = x.decode(input_code) #Decode
    except:
        u_string = x.decode('UTF-8')
    re.sub(r'\w{4}', u'xxxx', u_string, flags=re.UNICODE) #Handle funny unicode artifacts
    x = u_string.encode('UTF-8') #Encode back to UTF-8
    x_clean = str(x) #Convert to string

    if ner == True:
        x_clean = capitalize_named_entities(x_clean)

    if rem_stopwords == True:
        x_clean = remove_stopwords(x_clean)

    if lemmatize == True:
        x_clean = lemmatize_text(x_clean)

    soup = BeautifulSoup(x_clean, 'html.parser')
    x_clean = soup.text

    if rem_punc == True:
       x_clean = remove_punctuation(x_clean)

    #tokens = word_tokenize(x_clrunean)
    return x_clean

if __name__ == '__main__':
    df10_allthreads = ProcessedData(files=['biology.csv', 'cooking.csv','crypto.csv','diy.csv','robotics.csv','travel.csv'], n_samples_per_file=10, rem_stopwords=True, triple_title=True, ner=True, rem_punc = True,lemmatize=True)

    df10_allthreads.data.to_pickle('pickle_jar/dataframes/allthreads_first10.p')

    with open(r"pickle_jar/post_counters/allthreads_first1000_ctxt_dict.p", "wb") as output_file:
        cPickle.dump(df10_allthreads_minimal_processing.word_counts, output_file)

    with open(r"pickle_jar/tag_counters/allthreads_first1000_tag_dict.p", "wb") as output_file:
        cPickle.dump(df10_allthreads_minimal_processing.tag_counts, output_file)

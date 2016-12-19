from gensim.models import doc2vec
from gensim.models import Word2Vec
import ftfy
import chardet
import re
import numpy as np
import pandas as pd


def cleanText(x):
    '''General purpose text cleaner that turns text into a list of tokens fit for Word2Vec and Doc2Vec
        INPUT: string;
        OUTPUT: list of tokens.'''

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
    x = str(x) #Convert to string
    
    x = re.sub(r'[\[\]\"\'\,]',' ', x) #Remove backslashes
    x = re.sub(r'[^\w\s]',' ',x) #Insert space after end of each sentence
    x = x.split() #split string to list
    return x

def word2vec(series, sample=1e-3, size=256,window=10, min_count=0, workers=4):
    '''INPUT: Pandas Series
       OUTPUT: Trained Word2Vec Model'''
    #Clean all of the text in the series
    sentence_lists = list(series.iloc[:].apply(lambda x: cleanText(x)))
    #Train a Word2Vec Model
    model = Word2Vec(sentence_lists, workers=workers, size=size, min_count=min_count, window=window,sample=sample)

    return model

def Doc2Vec(dataframe):
    '''INPUT: Pandas Dataframe
       OUTPUT: Trained Doc2Vec model'''

    #Clean all the posts in the dataframe and create a LabeledSentence generator with the post and tag data.
    docs = []
    for i in range(len(dataframe)):
        post = cleanText(dataframe['ttl_ctxt'].values[i])
        tags = dataframe['tags'].values[i]
        labeledsent = doc2vec.LabeledSentence(words=post, tags=tags)
        docs.append(labeledsent)
    #Train the Doc2Vec model with a list of generators.
    model = doc2vec.Doc2Vec(docs)
    return model

def makeFeatureVec(words, model, num_features):
    '''INPUT: List of tokens, a pretrained model, and the dimensionality of the embedding.
       OUTPUT: A feature vector that maps the tokens input to the embedding space. Output feature vector is the average of its individual word vectors.'''

    #Create blank numpy array to store feature vector coordinates.
    feature_vec = np.zeros((num_features,), dtype='float32')
    n_words = 0 #Tally count of words in input.

       #Loop over all the tokens from the input and extract their coordinates from the model if present in the model vocabulary.
    for w in words:
        if w in set(model.index2word):
            n_words += 1
            feature_vec = np.add(feature_vec, model[w]) #Add to feature_vec

    # Divide feature_vec by n_words to get a Naive average
    featureVec = np.divide(feature_vec,n_words)
    return featureVec

def getAvgFeatureVecs(docs, model, num_features):
    '''INPUT: List of lists of strings (stack exchange posts)
       OUTPUT: 2D numpy array with an feature vector (embedding coordinates) for each document in a dataset.'''

    docFeatureVecs = np.zeros((len(docs), num_features), dtype='float32')

    #make feature vectors for each doc in the dataset
    for i,d in enumerate(docs):
        docFeatureVecs[i] = makeFeatureVec(d, model, num_features)

    return docFeatureVecs

def find_tags(model, num_features, dataframe, post_index, n_neighbors):
    '''INPUT: trained gensim model, model dimensionality, a pandas dataframe, and index, and the number of nearby tags to look for.
       OUTPUT: A sorted list of tuples for words with the highest cosine similarity to a document indexed from a dataframe. Each tuple contains a word and a numeric value.'''

    #Each post has a list of lists of tokens (broken up by sentence)
    #Flatten the list of lists into one list:
    post_tokens = dataframe['ttl_ctxt'].values[post_index]
    
    #Preprocess the posts 
    clean_posts = cleanText(post_tokens)
    
    #Average the word vectors of all the words in the post, Naive Doc2Vec
    avg_vec = makeFeatureVec(clean_posts, model, num_features)
    
    #Find the most similar word vectors by cosine similarity
    tag_candidates = model.similar_by_vector(avg_vec, topn=n_neighbors)
    
    return tag_candidates

def predict_tags(tag_candidates, dataframe, tag_counter):
    '''Input: tag_candidates(list of tuples), a pandas dataframe, and a Counter dictionary of tag frequencies in the corpus.
       Output: Top 5 plausible tags by cosine similarity and membership in the tag_counter object.'''
    tags = [] #empty list to hold tags
    tag_count = 0 #tag counter
    while tag_count <= 5: # Limit to 5 tags
        for i in range(len(tag_candidates)): #Loop through candidates
            if tag_candidates[i][0] in tag_counter.keys(): #if candidate is an explicity tag..
                if tag_candidates[i][1] > 0.2: #if the candidate has a cosine similarity greater than 0.2
                    tags.append(tag_candidates[i][0]) #append to tags list
                    tag_count += 1
                else:
                    pass

    return tags[:5]
    
def auto_tag(model, num_features, dataframe, n_neighbors, tag_counter):
    '''INPUT: a trained gensim model, dimensionality of the model, a pandas dataframe, number of neighbors to search for in embedding space, and a Counter object of tag frequencies in the corpus. '''

    predictions_vec = [] #bucket for holding predictions
    #loop through rows/docs in the dataframe
    for i in range(len(dataframe)):
        tag_candidates = find_tags(model, num_features, dataframe, i, n_neighbors) #Find n number of closest neighbors in embedding
        prediction = predict_tags(tag_candidates, dataframe, tag_counter) #Predict top tags for doc
        predictions_vec.append(prediction)
    return predictions_vec
    






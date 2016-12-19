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
from gensim.models import Doc2Vec
import logging
from featurize import Corpus_Features
import re
import chardet
import ftfy
from gensim.models import Word2Vec
from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm

from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

def kfold_evaluation(X,y,model, kf):
    """Performs k-fold cross validation on a dataset(X,y) with a given model. Stores model results in a system of dictionaries, which can be aggregated by mean and visualized in a pandas dataframe."""

    #Create defaultdict container structure
    kfold_performance = defaultdict(lambda: defaultdict(list))
    
    #for each round of training, apply train-test split indices to the data
    for k, (train, validate) in enumerate(kf.split(X)):
        #train model on training set
        model.fit(X[train], y[train])
        
        #Generate predtions on train and test splits
        predictions_train = model.predict(X[train])
        predictions_validate = model.predict(X[validate])

        #Calculate accuracy scores and store in container
        accuracy = accuracy_score(y[train], predictions_train)
        accuracy_validate = accuracy_score(y[validate], predictions_validate)
        kfold_performance['accuracy']['train'].append(accuracy)
        kfold_performance['accuracy']['validate'].append(accuracy_validate)

        #Calculate precision scores and store in container
        precision_train = precision_score(y[train], predictions_train, average='macro')
        precision_validate = precision_score(y[validate], predictions_validate, average='macro')
        kfold_performance['precision']['train'].append(precision_train)
        kfold_performance['precision']['validate'].append(precision_validate)

        #Calculate recall scores and store in container
        recall_train = recall_score(y[train], predictions_train, average='macro')
        recall_validate = recall_score(y[validate], predictions_validate, average='macro')
        kfold_performance['recall']['train'].append(recall_train)
        kfold_performance['recall']['validate'].append(recall_validate)

        #Calculate f1 scores and store in container
        f1_train = f1_score(y[train], predictions_train, average='macro')
        f1_validate = f1_score(y[validate], predictions_validate, average='macro')
        kfold_performance['f1']['train'].append(f1_train)
        kfold_performance['f1']['validate'].append(f1_validate)
        
    #Average performance metrics across each kfold
    for metric in kfold_performance.keys():
        for split in kfold_performance[metric].keys():
            kfold_performance[metric][split] = np.mean(kfold_performance[metric][split])
    
    #Output is ready to be read into pandas via panda.DataFrame.from_dict()        
    return kfold_performance

def stage_score_plot(model, error_metric):
    """Function to plot the error of a boosting model over boosting iterations. Model input must be pretrained."""
    #Extract staged prediction generators
    stage_train_predictions = model.staged_predict(x_train)
    stage_test_predictions = model.staged_predict(x_test)
    error_train_data = []
    error_test_data = []
    
    for _ in range(model.n_estimators):

        error_train = error_metric(y_train,next(stage_train_predictions))
        error_test =  error_metric(y_test,next(stage_test_predictions))
    
        error_train_data.append(error_train)
        error_test_data.append(error_test)
        
    #Create plot of boosting progress
    boost_iteration = [_ for _ in range(1,model.n_estimators + 1)]
    pylab.figure()
    
    plt.plot(boost_iteration, error_train_data, 'b', label='training error, LR = {}'.format(model.learning_rate), linewidth=2)
    plt.plot(boost_iteration, error_test_data, 'r', label='testing error, LR = {}'.format(model.learning_rate), linewidth=2)

    plt.title('{} error at each stage of boosting \n'.format(model.__class__.__name__))
    plt.xlabel('Boosting Iteration')
    plt.ylabel(str(error_metric))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


class Classify_Thread():
    def __init__(self, X, y):
        """X can be a bag of words model, tfidf vector, or doc2vec. y must be numeric, mapping to a thread name"""

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, random_state=42)
        self.feature_importances = defaultdict()
        

    def MNB_Gridsearch(self):
        
        mnb_grid = {
                              'alpha': [0.1,0.5, 1.0, 2.0],
                              'fit_prior': [True, False],
                              }

        mnb_gridsearch = GridSearchCV(MultinomialNB(),
                                     mnb_grid,
                                     n_jobs=-1,
                                     verbose=True,
                                     scoring='f1_macro')
        
        mnb_gridsearch.fit(self.x_train, self.y_train)
        return mnb_gridsearch.best_estimator_
   
    def RandomForest(self):
        rf = RandomForestClassifier()
        rf.fit(self.x_train, self.y_train)
        return rf
    
    def Logistic_Regression_OneVAll(self):
        lr = SGDClassifier(loss='log',penalty='elasticnet')
        lr_onevrest = OneVsRestClassifier(lr, n_jobs=-1)
        lr_onevrest.fit(self.x_train, self.y_train)
        return lr_onevrest
    
    def SVC_OneVAll(self):
        svc = SGDClassifier(loss='hinge', penalty='l2')
        svc_onevrest = OneVsRestClassifier(svc, n_jobs=-1)
        svc_onevrest.fit(self.x_train, self.y_train)
        return svc_onevrest
        

    def RandomForest_Gridsearch(self):
        random_forest_grid = {'max_depth': [None, 6, 10, 3],
                              'max_features': ['sqrt', 'log2', None],
                              'n_estimators': [10, 15, 20],
                               'n_jobs' : [-1],
                              }

        rf_gridsearch = GridSearchCV(RandomForestClassifier(),
                                     random_forest_grid,
                                     n_jobs=-1,
                                     verbose=True,
                                     scoring='f1_macro')
        
        rf_gridsearch.fit(self.x_train, self.y_train)

        return rf_gridsearch.best_estimator_

    
    def Gradient_Boosted_Trees(self,n_rows):
        gradient_boost = GradientBoostingClassifier(
                                 learning_rate=1.0,
                                 loss='deviance',
                                 n_estimators=20,
                                 max_depth = 1,
                                 )
        gradient_boost.fit(self.x_train[:n_rows], self.y_train[:n_rows])
        return gradient_boost
    
    def Adaboost(self,n_rows):
        adaboost = AdaBoostClassifier(DecisionTreeRegressor(),
                        learning_rate=1.0,
                        n_estimators=20,
                        )
        adaboost.fit(self.x_train[:n_rows], self.y_train[:n_rows])
        return adaboost
        
    def Gradient_Boosted_Trees_Gridsearch(self):
        gradient_boosting_grid = {
                              'learning_rate': [0.05, 0.1, 0.3],
                              'n_estimators': [50],
                              'max_depth': [1,3,7],
                              'min_samples_split': [2, 5, 10],
                              'min_samples_leaf': [1, 5, 10],
                              'subsample': [1, 0.5, 0.75],
                              'max_features': [None,'sqrt','log2'],
                              }


        gb_gridsearch = GridSearchCV(GradientBoostingRegressor(),
                                     gradient_boosting_grid,
                                     n_jobs=-1,
                                     verbose=True,
                                     scoring='neg_mean_squared_error')
        gb_gridsearch.fit(x_train, y_train)
        best_gb_model = gb_gridsearch.best_estimator_
        


if __name__ == '__main__':
    data_file = open("pickle_jar/dataframes/allthreads_first1000_data.p", 'rb')
    data = pickle.load(data_file)

    tag_token_counter_file = open('pickle_jar/tag_counters/allthreads_first1000_minimal_processing_tag_dict.p', 'rb')
    tag_token_counter = pickle.load(tag_token_counter_file)

    post_token_counter_file = open('pickle_jar/post_counters/allthreads_first1000_minimal_processing_ctxt_dict.p', 'rb')
    post_token_counter = pickle.load(post_token_counter_file)

    features = Corpus_Features(data, post_token_counter, tag_token_counter)
    features.tfidf()

    X = features.tfidf_vec.toarray()
    y = features.tags_in_post()
    x_train, x_test, y_train, y_test = train_test_split(X, y)

    # mnb = MultinomialNB()
    # mnb.fit(x_train, y_train)
    # pred = mnb.predict(x_test)
    # #score = mnb.score(y_test, pred)
    
    # n_correct, n_incorrect, correct, incorrect = error_analysis(pred, y_test)

    # cf_mtrx = confusion_matrix(y_test, pred)

    # mnb_ovr = OneVsRestClassifier(MultinomialNB(), n_jobs=-1)
    # mnb_ovr.fit(x_train, y_train)
    # pred = mnb_ovr.predict(x_test)
    # print(confusion_matrix(y_test, pred))
    # print(mnb_ovr.score(x_test,y_test))

    svc_multilabel = OneVsRestClassifier(svm.SVC(kernel='linear'))
    svc_multilabel.fit(x_train, y_train)
    svc.fit(X,y)









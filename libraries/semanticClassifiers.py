# The Document-topic model based naive Bayes classfication of reviews
#
# this class fits a document topic model, retains it as a property 
# of the object, and then fits an sklearn.naive_bayes classifier in
# that document-topic model feature space
#
# When called to predict this class retrieves the previously fit
# document-topic model, projects the training data into that space
# and applies the fitted sklearn.naive_bayes classifier to obtain
# predictions
# 
# You should be able to drop in any other sklearn estimator as a parent
# class to adapt this class to other classifiers

from sklearn.utils.validation import check_is_fitted

from gensim.models import LsiModel as lsi
from gensim.corpora import Dictionary
from gensim import matutils

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

class semantic_NB(GaussianNB):
    semSpace=[]
    this_dict = []
    
    # reviews should be a list of reviews, where each review is itself a 'cleaned'
    # list of words (lematized, no stop words, etc). train_lbls should be a
    # boolean array
    def fit(self, train_reviews, train_lbls):
        # train a document-topic model        
        self.this_dict = Dictionary(train_reviews)

        # transform corpus (train) into a 2d array word counts (a 'bag of words')
        bow_corpus = [self.this_dict.doc2bow(text) for text in train_reviews]

        # construct a semantic model based on document-topic similarity (15-20 min for 1500k reviews?)
        self.semSpace = lsi(bow_corpus, id2word=self.this_dict, num_topics=300, chunksize=100000, distributed=False)

        # Apply the semantic model to the training set bag of words (fast)
        feat_train = self.semSpace[bow_corpus]

        # convert from TransformedCorpus datatype to numpy doc x topic array (medium speed, needs more benchmarking)
        train_topics_csr = matutils.corpus2csc(feat_train)
        feat_train_np = train_topics_csr.T.toarray()
        
        # fit naive bayes model to training features and apply it to test features
        return super().fit(feat_train_np, train_lbls)
    
    def predict(self, test_reviews):   
        check_is_fitted(self)
        
        # Apply semantic model to test set
        bow_corpus_test = [self.this_dict.doc2bow(text) for text in test_reviews]
        feat_test = self.semSpace[bow_corpus_test]
        test_topics_csr = matutils.corpus2csc(feat_test)
        feat_test_np = test_topics_csr.T.toarray()

        return super().predict(feat_test_np)
    
    
class semantic_SVM(SVC):
    semSpace=[]
    this_dict = []
    
    # reviews should be a list of reviews, where each review is itself a 'cleaned'
    # list of words (lematized, no stop words, etc). train_lbls should be a
    # boolean array
    def fit(self, train_reviews, train_lbls):
        # train a document-topic model        
        self.this_dict = Dictionary(train_reviews)

        # transform corpus (train) into a 2d array word counts (a 'bag of words')
        bow_corpus = [self.this_dict.doc2bow(text) for text in train_reviews]

        # construct a semantic model based on document-topic similarity (15-20 min for 1500k reviews?)
        self.semSpace = lsi(bow_corpus, id2word=self.this_dict, num_topics=300, chunksize=100000, distributed=False)

        # Apply the semantic model to the training set bag of words (fast)
        feat_train = self.semSpace[bow_corpus]

        # convert from TransformedCorpus datatype to numpy doc x topic array (medium speed, needs more benchmarking)
        train_topics_csr = matutils.corpus2csc(feat_train)
        feat_train_np = train_topics_csr.T.toarray()
        
        # fit naive bayes model to training features and apply it to test features
        return super().fit(feat_train_np, train_lbls)
    
    def predict(self, test_reviews):   
        check_is_fitted(self)
        
        # Apply semantic model to test set
        bow_corpus_test = [self.this_dict.doc2bow(text) for text in test_reviews]
        feat_test = self.semSpace[bow_corpus_test]
        test_topics_csr = matutils.corpus2csc(feat_test)
        feat_test_np = test_topics_csr.T.toarray()

        return super().predict(feat_test_np)
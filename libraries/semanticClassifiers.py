# a transformer method for use in sklearn pipelines when 
# evaluating bow in a latent semantic space

import numpy as np

from gensim.models import LsiModel as lsi
from gensim.corpora import Dictionary
from gensim import matutils

from sklearn.base import TransformerMixin, BaseEstimator

# this is a class to accommodate semantic space mappings. It takes
# a bow representation as input and returns features in a latent
# semantic space as output
#
# The class is a valid sklearn transformer and can be used as such
# in sklearn pipelines. For details refer to,
# https://scikit-learn.org/stable/modules/compose.html
#
# Also, this is another useful reference,
# https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html
class docTopTransformer(TransformerMixin, BaseEstimator):
    
    def __init__(self, this_dict=None, d=300, distributed=False):
        self.this_dict = this_dict
        self.d = d
        self.distributed = distributed
        
    def fit(self, X, y=None):
        corpus = matutils.Dense2Corpus(np.transpose(X))
        
        # construct a semantic model based on document-topic similarity (15-20 min for 1500k reviews?)
        self.semSpace = lsi(corpus, id2word=self.this_dict, num_topics=self.d, 
                            chunksize=10000, distributed=self.distributed)
        
        return self
    
    def transform(self, X, y=None):
        corpus = matutils.Dense2Corpus(np.transpose(X))
        
        # Apply the semantic model to the training set bag of words (fast)
        feat = self.semSpace[corpus]

        # convert from TransformedCorpus datatype to numpy doc x topic array (medium speed, needs more benchmarking)
        topics_csr = matutils.corpus2csc(feat)
        X_ = topics_csr.T.toarray()
        
        return X_

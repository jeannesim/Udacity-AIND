import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        bestmodel = self.base_model(self.n_constant)
        bestscore = float('inf')
        n_features = len(self.X[0])

        try:
            for num_components in range(self.min_n_components, self.max_n_components+1):
                currentmodel = GaussianHMM(num_components=num_components, random_state=self.random_state, n_iter=1000).fit(self.X, self.lengths)
                logL = currentmodel.score(self.X, self.lengths)
                p = num_components * num_components + 2 * n_features*num_components - 1
                N = len(self.X)
                currentscore = (-2)*logL+p*np.log(N)
                if currentscore < bestscore:
                    bestmodel = currentmodel
                    bestscore = currentscore

        except Exception as e:
            pass

        return bestmodel


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        bestmodel = self.base_model(self.n_constant)
        bestscore = float('-inf')
        try:

            for num_components in range(self.min_n_components, self.max_n_components+1):
                currentmodel = GaussianHMM(num_components=num_components, random_state=self.random_state, n_iter=1000).fit(self.X, self.lengths)
                logL = currentmodel.score(self.X, self.lengths)

                sum_logL = []
                for word in self.words:
                    if word == self.this_word:
                        continue
                    word_n, word_len = self.hwords[word]

                    sum_logL.append(currentmodel.score(word_n, word_len))

                currentscore = logL - np.average(sum_logL)
                if currentscore > bestscore:
                    bestmodel = currentmodel
                    bestscore = currentscore
        except Exception as e:
            pass

        return bestmodel
        
class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        bestmodel = self.base_model(self.n_constant)
        bestscore = float('-inf')
        try:        
            for num_components in range(self.min_n_components, self.max_n_components+1):
                split_method = KFold(n_splits=min(3,len(self.lengths)))
                logL = []
                for train_idx, test_idx in split_method.split(self.sequences):
                    train_X, train_length = combine_sequences(train_idx, self.sequences)
                    test_X, test_length = combine_sequences(test_idx, self.sequences)
                    currentmodel = GaussianHMM(num_components=num_components, random_state=self.random_state, n_iter=1000).fit(train_X, train_length)
                    logL.append(currentmodel.score(test_X, test_length))
                currentscore = np.average(logL)
                if currentscore > bestscore:
                    bestmodel = currentmodel
                    bestscore = currentscore
                    
        except Exception as e:
            pass

        return bestmodel
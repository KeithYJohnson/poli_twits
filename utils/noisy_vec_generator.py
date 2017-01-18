from scipy.stats import rv_discrete
import numpy as np
from ipdb import set_trace as st

class NoisyVecGenerator():
    def __init__(self, df, words_dict):
        # Inputs:
        # - df: dataframe where each row is a word in the order it appears in the original corpus
        #       This is not single instance of each token, but however many times each one occurs
        # - words_dict: key: a word, value, the word vector it maps to
        self.df     = df
        self.words_dict = words_dict

        # An descending ordered Series. The index is the word, the value is how many times it occurs.
        self.values = df.body.value_counts()
        self.create_distribution()

    def sample(self):
        sampled_index = self.dist.rvs()
        word = self.values.index[sampled_index]
        words_vector_index = self.words_dict[word]

        return words_vector_index, word

    def create_distribution(self):
        num_unique = len(self.words_dict)
        num_train  = sum(self.values)

        xk = np.arange(num_unique)
        pk = (self.values.values / num_train)
        # TODO Implement this particular distribution from http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
        # We investigated a  of choices for Pn(w) and found that the
        # unigram distribution U(w) raised to the 3/4rd power (i.e.,
        # U(w)3/4/Z) outperformed significantly the unigram and the uniform distributions, for both NCE
        # and NEG on every task we tried including language modeling (not reported here).

        self.dist = rv_discrete(name='custm',values=(xk, pk))

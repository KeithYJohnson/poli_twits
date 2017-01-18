from scipy.stats import rv_discrete
import numpy as np

class NoisyVecGenerator():
    def __init__(self, df):
        self.df     = df
        self.values = df.body.value_counts()
        self.words  = self.values.index
        self.create_distribution()

    def sample(self):
        sampled_index = self.dist.rvs()
        word = self.words[sampled_index]

        # TODO just realized I have multiple word vectors for the same word.
        df_index = self.df.body[self.df.body == word].index.tolist()[0]
        return df_index, word

    def create_distribution(self):
        num_unique = len(self.values)
        num_train  = sum(self.values)

        xk = np.arange(num_unique)
        pk = (self.values.values / num_train)
        
        # TODO Implement this particular distribution from http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
        # We investigated a  of choices for Pn(w) and found that the
        # unigram distribution U(w) raised to the 3/4rd power (i.e.,
        # U(w)3/4/Z) outperformed significantly the unigram and the uniform distributions, for both NCE
        # and NEG on every task we tried including language modeling (not reported here).

        self.dist = rv_discrete(name='custm',values=(xk, pk))

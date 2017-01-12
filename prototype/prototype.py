# Just seeing how well gensim works and
# what the doc vectors actually look like
from ipdb import set_trace as st
import pandas as pd
import numpy as np
from math import floor
from gensim.models import Doc2Vec
from labeled_line_sentences import LabeledLineSentence
from sklearn.linear_model import LogisticRegression

keyword_tweets_df = pd.read_csv(
    './../data/keyword_tweets.txt',
    header=None,
    index_col=False,
    names=['is_political', 'body'],
)

general_tweets_df = pd.read_csv(
    './../data/general_tweets.txt',
    header=None,
    index_col=False,
    names=['is_political', 'body'],
)

all_tweets_df = pd.read_csv(
    './../data/all_tweets.txt',
    header=None,
    index_col=False,
    names=['is_political', 'body'],
)

# LabeledLineSentence expects only the body
# separated by line breaks, which is what these
# files contain
keyword_tweet_file = 'keyword_tweets_bodies.txt'
general_tweet_file = 'general_tweets_bodies.txt'
all_tweets_file = 'all_tweets_file.txt'

# Relative file system location of the models.
keyword_tweet_d2v_file = 'keyword_tweets.d2v'
all_tweets_d2v_file = 'all_tweets2.d2v'

sources = {
    all_tweets_file: 'all_tweets'
}


embedding_size = 100
model = None
try:
    model = Doc2Vec.load(all_tweets_d2v_file)
except FileNotFoundError:
    print('Couldnt find a doc2Vec model for {}, gonna train one for ya instead.'.format(all_tweets_d2v_file))

if not model:
    sentences = LabeledLineSentence(sources)
    model = Doc2Vec(min_count=1, window=10, size=embedding_size, sample=1e-4, negative=5, workers=8)
    model.build_vocab(sentences.to_array())
    for epoch in range(10000):
        print('epoch: ', epoch)
        model.train(sentences.sentences_perm())

    model.save('./{}'.format(keyword_tweet_d2v_file))


shuffled_df = all_tweets_df.sample(frac=1)
train_test_percent_split = .8
train_test_index_split = floor(shuffled_df.shape[0] * train_test_percent_split)

all_embeddings   = np.array(model.docvecs)
train_embeddings = all_embeddings[shuffled_df.index][0:train_test_index_split, :]
test_embeddings  = all_embeddings[shuffled_df.index][train_test_index_split:, :]

classifier = LogisticRegression()
classifier.fit(train_embeddings, shuffled_df.is_political[0:train_test_index_split])

print('score on training data: ', classifier.score(train_embeddings, shuffled_df.is_political[0:train_test_index_split]))
print('score on test data: ', classifier.score(test_embeddings,  shuffled_df.is_political[train_test_index_split:]))

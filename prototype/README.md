So this is just a quick prototype I rigged up
to play around with paragraph vectors, aka doc2vec, described
in this [paper](https://cs.stanford.edu/~quocle/paragraph_vector.pdf) by Quoc Le and Tomas Mikolov

It uses the gensim library's [doc2Vec model](https://radimrehurek.com/gensim/models/doc2vec.html)

I'll be coding up a political sentiment classifier without gensim soon.  This prototype just
detects whether there is some agenda but I aim to additionally classify where on the
political spectrum a given tweet lands.

## Data

I used the data from [The Twitter Political Corpus](https://www.usna.edu/Users/cs/nchamber/data/twitter/)
This data was used by Micol Marchetti-Bowick and Nathanael Chambers to publish [this paper](http://anthology.aclweb.org/E/E12/E12-1062.pdf)

## Results

After 1000 epochs

```python
print('score on training data: ', classifier.score(train_embeddings, shuffled_df.is_political[0:train_test_index_split]))
score on training data:  0.78081321474
print('score on test data: ', classifier.score(test_embeddings,  shuffled_df.is_political[train_test_index_split:]))
score on test data:  0.775095298602
```

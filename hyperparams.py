num_features = 100
batch_size   = 50
window_size  = 5 # How many words on both the left and right of the target word to include in the context.
num_noise    = 2
filename = 'cleaned_data/cleanedwords_from_all_tweets.csv'

import os, re
string = re.findall(r'\d+', os.popen('wc -l {}'.format(filename)).read())
num_train = int(string[0])

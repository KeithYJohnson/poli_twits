import pandas as pd
words_df = pd.read_csv(
    'cleaned_data/cleanedwords_from_all_tweets.csv',
    header=None,
    index_col=False,
    names=['body']
)

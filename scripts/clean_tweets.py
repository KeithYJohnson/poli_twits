from utils.data_cleaner import DataCleaner

cleaner = DataCleaner('data/all_tweets.txt','body', header_names=['is_political', 'body'])
cleaner.clean()

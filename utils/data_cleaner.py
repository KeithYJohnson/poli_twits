from bs4 import BeautifulSoup
import re
import pandas as pd
import csv
from nltk.corpus import stopwords
from os import path
from spellcheckers import no_three_letters_in_row

MINIMUM_TWEET_WORD_LENGTH = 5

class DataCleaner:
    def __init__(self, filename, column_to_clean, has_header=None, delimiter=",", header_names=[]):
        self.filename = filename
        self.delimiter = delimiter
        self.df = pd.read_csv(filename, header=has_header, delimiter=",", names=header_names)
        self.column_to_clean = column_to_clean

    def clean(self):
        self.cleaned_data = self.df[self.column_to_clean].map(lambda x: self.clean_words(x))
        self.cleaned_data = self.cleaned_data.dropna()
        self.save_data()

    def save_data(self):
        basename = path.basename(self.filename)
        no_ext  =  path.splitext(basename)[0]
        cleaned_data_filename = "cleanedtext_of_{}.csv".format(no_ext)

        cleaned_data_file = open(
            path.dirname(__file__) + \
            '/../cleaned_data/{}'.format(cleaned_data_filename), 'w+'
        )
        self.cleaned_data.to_csv(cleaned_data_file)

        with open(path.dirname(__file__) + '/../cleaned_data/cleanedwords_from_{}.csv'.format(no_ext), 'w+') as f:
            csv_writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            self.cleaned_data.map(
                lambda sentence: csv_writer.writerows(
                    #[[word]] to give writerows a sequence of one word instead of a sequence of len(word) characters
                    [[word] for word in sentence.split()]
                )
            )

    def clean_words(self, words):
        text = BeautifulSoup(words, 'html5lib').get_text()
        # Replace all URLs with <URL>
        text = re.sub(r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
                      '<URL>', text)
        # Remove non-letters
        # TODO - what about hyphenated words?
        text = re.sub("(?!<URL>)([^a-zA-Z])", " ", text)

        #TODO - <URL> token actually becomes <url, it probably doesnt matter as far as
        # building word vectors goes, but at some point make it return <URL>
        words = text.lower().split()

        stops = set(stopwords.words("english"))
        stops.add('rt') # Add retween acronym to stopwords
        #Remove stop words and reduced all instances of three consecutive characters to two consecutive characters.
        words = [no_three_letters_in_row(w) for w in words if not w in stops]
        if len(words) > MINIMUM_TWEET_WORD_LENGTH:
            return( " ".join( words ))

if __name__ == '__main__':
    cleaner = DataCleaner('data/all_tweets.txt','body', header_names=['is_political', 'body'])
    cleaner.clean()

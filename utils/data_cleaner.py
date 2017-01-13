from bs4 import BeautifulSoup
import re
import pandas as pd
from nltk.corpus import stopwords
from os import path

class DataCleaner:
    def __init__(self, filename, column_to_clean, has_header=None, delimiter=",", header_names=[]):
        self.filename = filename
        self.delimiter = delimiter
        self.df = pd.read_csv(filename, header=has_header, delimiter=",", names=header_names)
        self.column_to_clean = column_to_clean

    def clean(self):
        self.cleaned_data = self.df[self.column_to_clean].map(lambda x: self.clean_words(x))
        self.save_data()

    def save_data(self):
        basename = path.basename(self.filename)
        no_ext  =  path.splitext(basename)[0]
        cleaned_data_filename = "cleanedtext_of_{}.csv".format(no_ext)

        print('cleaned_data_filename: ', cleaned_data_filename)
        cleaned_data_file = open(
            path.dirname(__file__) + \
            '/../cleaned_data/{}'.format(cleaned_data_filename), 'w+'
        )
        self.cleaned_data.to_csv(cleaned_data_file)

    def clean_words(self, words):
        review_text = BeautifulSoup(words, 'html5lib').get_text()
        # Remove non-letters
        review_text = re.sub("[^a-zA-Z]", " ", review_text)
        words = review_text.lower().split()

        #Remove stop wrds
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

        return( " ".join( words ))
import csv
import numpy as np
from sklearn import preprocessing
import re
import pandas as pd
import text_preprocess
import sys
csv.field_size_limit(sys.maxsize)


def load_ccat_data(data_path):
    # load data (can be used if data already split into train and test set)
    data = pd.read_csv(data_path, header=0)
    x = np.array(data['article'])
    y = np.array(data['class'])
    x_mapped = text_preprocess.mapped_text(x)  # mapping with listed character
    print x_mapped
    # transform y (label of author name) into integer label (start from 1)
    auth_class = list(set(y))
    print auth_class
    le = preprocessing.LabelEncoder()
    le.fit(auth_class)
    y_numeric = le.transform(y)
    return np.array(x_mapped), np.array(y_numeric)


def load_data_imdb62(data_path_imdb):
    author = []
    content = []
    with open(data_path_imdb, 'r') as f:
        for line in f:
            per_line = line.split("\t")
            author.append(per_line[0])
            content.append(per_line[1])
        # x = np.array(content)
        y = np.array(author)
    f.close()
    x_mapped = text_preprocess.mapped_text(content)
    # encode labels with value between 0 and n_author-1
    auth_class = list(set(y))
    le = preprocessing.LabelEncoder()
    le.fit(auth_class)
    y_numeric = le.transform(y)
    return np.array(x_mapped), np.array(y_numeric)


def load_data_judgment(data_judgment):
    with open(data_judgment, 'rb') as f:
        reading = csv.reader(f, delimiter='\t')
        author = []
        content = []
        for row in reading:
            auth = row[1].split(".")
            auth_class = re.sub(r"[^A-Za-z]+", '', auth[0])
            if row[0].lower() == "rich1913-1928" or row[0].lower() == "dixon" or row[0].lower() == "mctiernan1965-1975":
                author.append(auth_class.lower())
                content.append(row[2])
        x = np.array(content)
        y = np.array(author)
        x_mapped = text_preprocess.mapped_text(x)  # mapping with listed character

    # transform y (label of author name) into integer label (start from 1)
    auth_class = list(set(y))
    print auth_class
    le = preprocessing.LabelEncoder()
    le.fit(auth_class)
    y_numeric = le.transform(y)
    return np.array(x_mapped), np.array(y_numeric)

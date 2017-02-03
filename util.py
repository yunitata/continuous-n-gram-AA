'''
Much of the code is modified from
https://github.com/fchollet/keras/blob/master/examples/imdb_fasttext.py
'''
from __future__ import print_function
import numpy as np
from nltk.tokenize import word_tokenize
from keras.preprocessing import sequence
from sklearn.metrics import accuracy_score
np.random.seed(1337)


def turn_to_integer(sequences, indices, mode):
    new_sequences = []
    for input_list in sequences:
        new_list = []
        if mode == 'word':
            for word in word_tokenize(input_list):
                if word in indices:
                    new_list.append(indices[word])
                else:
                    new_list.append(indices["UNK"])
            new_sequences.append(new_list)
        else:
            for character in input_list:
                if character in indices:
                    new_list.append(indices[character])
            new_sequences.append(new_list)
    return new_sequences


def create_ngram_set(input_list, ngram_value):
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range, mode):
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        ngram_list = []
        if mode == 'word':
            ngram_list = new_list
        for i in range(len(new_list)-ngram_range+1):
            for ngram_value in range(2, ngram_range+1):
                ngram = tuple(new_list[i:i+ngram_value])
                if ngram in token_indice:
                    ngram_list.append(token_indice[ngram])
        new_sequences.append(ngram_list)
    return new_sequences


def input_rep(ngram_range, indices, maxlen, tr, vl, ts, mode):
    tr = turn_to_integer(tr, indices, mode)
    vl = turn_to_integer(vl, indices, mode)
    ts = turn_to_integer(ts, indices, mode)
    max_features = len(indices)
    if ngram_range > 1:
        print('Adding {}-gram features'.format(ngram_range))
        ngram_set = set()
        for input_list in tr:
            for i in range(2, ngram_range+1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)
        start_index = max_features + 1
        token_indice = {v: k+start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}
        max_features = np.max(list(indice_token.keys())) + 1
        tr = add_ngram(tr, token_indice, ngram_range, mode)
        vl = add_ngram(vl, token_indice, ngram_range, mode)
        ts = add_ngram(ts, token_indice, ngram_range, mode)
    print('Pad sequences (samples x time)')
    tr = sequence.pad_sequences(tr, maxlen=maxlen)
    vl = sequence.pad_sequences(vl, maxlen=maxlen)
    ts = sequence.pad_sequences(ts, maxlen=maxlen)
    return tr, vl, ts, max_features


def eval(y_pred, y_ts):
    acc = accuracy_score(y_pred, y_ts)
    return acc

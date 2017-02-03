import sys
import load_data
import numpy as np
from sklearn.cross_validation import StratifiedKFold, train_test_split
import util
import model
import text_preprocess
from sklearn.feature_extraction.text import CountVectorizer
from keras.utils.np_utils import to_categorical
import logging
np.random.seed(1337)


def train_test(x_tr, x_val, x_ts, y_tr, y_val, y_ts,
               mode, batch_size, nb_class, nb_epoch,
               seq_length_word, seq_length_char, lr):
    # turn the class label into categorical
    y_tr = to_categorical(y_tr)
    y_val = to_categorical(y_val)
    y_ts = to_categorical(y_ts)

    # create vocabulary for word
    word_vectorizer = CountVectorizer(analyzer='word', max_features=700, max_df=0.95, min_df=2, ngram_range=(1,1))
    wrd = word_vectorizer.fit(x_tr)
    vocab = wrd.vocabulary_.keys()
    vocab.append('UNK')
    word_indices = dict((c, i) for i, c in enumerate(vocab))

    # create vocabulary for character
    chars = text_preprocess.character_list
    char_indices = dict((c, i) for i, c in enumerate(chars))

    acc = 0
    ngram_range_word = 2
    ngram_range_char = 4

    # represent the input and train test the model
    if mode == 'word':
        train, valid, test, max_features = util.input_rep(ngram_range_word, word_indices, seq_length_word, x_tr, x_val, x_ts, mode)
        acc = model.model_woc(train, valid, test, y_tr, y_val, y_ts, max_features, nb_class, emb_size, seq_length_word, nb_epoch, batch_size, lr)

    elif mode == 'char':
        train, valid, test, max_features = util.input_rep(ngram_range_char, char_indices, seq_length_char, x_tr, x_val, x_ts, mode)
        acc = model.model_woc(train, valid, test, y_tr, y_val, y_ts, max_features, nb_class, emb_size, seq_length_char, nb_epoch, batch_size, lr)

    elif mode == 'wordchar':
        train_word, valid_word, test_word, max_features_word = util.input_rep(ngram_range_word, word_indices, seq_length_word, x_tr, x_val, x_ts, "word")
        train_char, valid_char, test_char, max_features_char = util.input_rep(ngram_range_char, char_indices, seq_length_char, x_tr, x_val, x_ts, "char")
        acc = model.model_wac(train_char, valid_char, test_char, train_word, valid_word, test_word,
                              y_tr, y_val, y_ts, max_features_char, max_features_word,
                              nb_class, emb_size, seq_length_word, seq_length_char, nb_epoch, batch_size, lr)
    return acc


# example command
# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python training_testing.py -datasetname -mode -trainfilepath -testfilepath
# add detail of each dataset (mode, length of text, number of epoch, embedding size, batch size, number of class etc)

if __name__ == "__main__":
    data_set = sys.argv[1]
    mode = sys.argv[2]
# create logging file
    if mode == "word":
        logfile = "WordModel.out"
        logging.basicConfig(filename=logfile, filemode='a', level=logging.DEBUG)
    elif mode == "char":
        logfile = "charModel.out"
        logging.basicConfig(filename=logfile, filemode='a', level=logging.DEBUG)
    else:
        logfile = "WordChar.out"
        logging.basicConfig(filename=logfile, filemode='a', level=logging.DEBUG)

    if len(sys.argv) == 5:
        batch_size = 5
        emb_size = 100
        nb_epoch = 150
        seq_length_word = 1000
        seq_length_char = 9000
        lr = 0.001

        file_train = sys.argv[3]
        file_test = sys.argv[4]
        x_train, y_train = load_data.load_ccat_data(file_train)
        x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
        x_ts, y_ts = load_data.load_ccat_data(file_test)
        if data_set == 'ccat10':
            nb_class = 10
            acc = train_test(x_tr, x_val, x_ts, y_tr, y_val, y_ts, mode, batch_size, nb_class, nb_epoch,
                             seq_length_word, seq_length_char, lr)
            print "Test Accuracy:", acc
            logging.debug(acc)
        elif data_set == 'ccat50':
            nb_class = 50
            acc = train_test(x_tr, x_val, x_ts, y_tr, y_val, y_ts, mode, batch_size, nb_class, nb_epoch,
                             seq_length_word, seq_length_char, lr)
            print "Test Accuracy:", acc
            logging.debug(acc)
    elif len(sys.argv) == 4 and data_set == 'judgment':
        nb_class = 3
        batch_size = 5
        emb_size = 100
        nb_epoch = 150
        seq_length_word = 4000
        seq_length_char = 30000
        lr = 0.001
        results = []
        file_path = sys.argv[3]
        X, Y = load_data.load_data_judgment(file_path)
        skf = StratifiedKFold(Y, n_folds=10, random_state=42, shuffle=True)
        i = 0
        for train_index, test_index in skf:
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            x_tr, x_val, y_tr, y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)
            acc = train_test(x_tr, x_val, X_test, y_tr, y_val, Y_test, mode, batch_size, nb_class, nb_epoch,
                             seq_length_word, seq_length_char, lr)
            print "cv: " + str(i) + ", acc: " + str(acc)
            i += 1
            results.append(acc)
            logging.debug(acc)
        print "Test Accuracy:", str(np.mean(results))
        #logging.debug(results)
    elif len(sys.argv) == 4 and data_set == 'imdb':
        nb_class = 62
        batch_size = 32
        emb_size = 50
        nb_epoch = 20
        seq_length_word = 600
        seq_length_char = 4000
        lr = 0.01
        results = []
        file_path = sys.argv[3]
        X, Y = load_data.load_data_imdb62(file_path)
        skf = StratifiedKFold(Y, n_folds=10, random_state=42, shuffle=True)
        i = 0
        for train_index, test_index in skf:
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            x_tr, x_val, y_tr, y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)
            acc = train_test(x_tr, x_val, X_test, y_tr, y_val, Y_test, mode, batch_size, nb_class, nb_epoch,
                             seq_length_word, seq_length_char, lr)
            print "cv: " + str(i) + ", acc: " + str(acc)
            i += 1
            results.append(acc)
            logging.debug(acc)
        print "Test Accuracy:", str(np.mean(results))
        #logging.debug(results)

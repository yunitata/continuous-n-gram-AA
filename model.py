'''
Much of the code is modified from
https://github.com/fchollet/keras/blob/master/examples/imdb_fasttext.py
'''

from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Merge, Embedding, AveragePooling1D
import keras.callbacks
from keras.optimizers import Adam
np.random.seed(1337)


def model_woc(train, 
              valid, 
              test, 
              y_tr, 
              y_val, 
              y_test, 
              max_features, 
              nb_class, 
              emb_size, 
              seq_length, 
              nb_epoch, 
              batch_size, 
              lr):

    model = Sequential()
    model.add(Embedding(max_features, output_dim=emb_size, input_length=seq_length, dropout=0.75, init='glorot_uniform'))
    model.add(AveragePooling1D(pool_length=model.output_shape[1]))
    model.add(Flatten())
    model.add(Dense(nb_class, activation='softmax'))
    adam = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='auto')
    print ("Training the model...")
    model.fit(train, y_tr,
                  nb_epoch=nb_epoch,
                  batch_size=batch_size,
                  validation_data=[valid, y_val],
                  callbacks=[earlystop_cb])
    print ("Evaluate on test data..")
    loss, acc = model.evaluate(test, y_test, verbose=2)
    #pred = model.predict_classes(test)
    return acc


def model_wac(train_char, 
              valid_char, 
              test_char, 
              train_wd, 
              valid_wd, 
              test_wd, 
              y_tr, 
              y_val, 
              y_ts, 
              max_features_char, 
              max_features_wd,
              nb_class, 
              emb_size,
              seq_length_word, 
              seq_length_char, 
              nb_epoch, 
              batch_size, 
              lr):
    modelw = Sequential()
    modelw.add(Embedding(max_features_wd, emb_size, input_length=seq_length_word, dropout=0.75, init='glorot_uniform'))
    modelw.add(AveragePooling1D(pool_length=modelw.output_shape[1]))
    modelw.add(Flatten())

    modelc = Sequential()
    modelc.add(Embedding(max_features_char, emb_size, input_length=seq_length_char, dropout=0.75, init='glorot_uniform'))
    modelc.add(AveragePooling1D(pool_length=modelc.output_shape[1]))
    modelc.add(Flatten())

    model = Sequential()
    model.add(Merge([modelw, modelc], mode='max'))
    model.add(Dense(nb_class, activation='softmax'))
    adam = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='auto')
    print ("Training the model...")
    model.fit([train_wd, train_char], y_tr,
                      nb_epoch=nb_epoch,
                      batch_size=batch_size,
                      validation_data=([valid_wd,valid_char], y_val),
                      callbacks=[earlystop_cb])
    print ("Evaluate on test data...")
    loss, acc = model.evaluate([test_wd, test_char], y_ts)
    # pred = model.predict_classes([test_wd, test_char])
    return acc

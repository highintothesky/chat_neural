# train using https://pypi.org/project/keras-self-attention/
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re
import keras
import h5py
import numpy as np
from keras.utils import np_utils
from keras import metrics
from keras_self_attention import SeqSelfAttention
from keras_multi_head import MultiHead
from keras.utils import CustomObjectScope
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Dropout, GRU, GlobalMaxPool1D
from keras.layers import GlobalMaxPool2D, BatchNormalization, Add, Flatten
from keras.layers import TimeDistributed, Multiply, Concatenate, Bidirectional
from keras import backend as K
from batch_generator import BatchGenerator
from imblearn.keras import BalancedBatchGenerator

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def main():
    # load test data
    filename = "data/memes.txt"
    raw_text = open(filename, 'r', encoding='utf-8').read()
    raw_text = raw_text.lower()
    raw_text = re.sub('\n', " ", raw_text)

    # create mapping of unique chars to integers
    chars = sorted(list(set(raw_text)))
    # print('-> chars:', chars)
    # char_to_int = dict((c, i) for i, c in enumerate(chars))
    char_to_int = {' ': 0, '!': 1, '%': 2, '&': 3, "'": 4, ',': 5, '-': 6, '.': 7, '/': 8, '0': 9, '1': 10, '2': 11, '3': 12, '4': 13, '5': 14, '6': 15, '7': 16, '8': 17, '9': 18, ':': 19, ';': 20, '<': 21, '>': 22, '?': 23, 'a': 24, 'b': 25, 'c': 26, 'd': 27, 'e': 28, 'f': 29, 'g': 30, 'h': 31, 'i': 32, 'j': 33, 'k': 34, 'l': 35, 'm': 36, 'n': 37, 'o': 38, 'p': 39, 'q': 40, 'r': 41, 's': 42, 't': 43, 'u': 44, 'v': 45, 'w': 46, 'x': 47, 'y': 48, 'z': 49, '~': 50, '—': 51}
    char_list = list(char_to_int.keys())
    raw_text = ''.join([i for i in raw_text if i in char_list])
    print('-> char to int:', char_to_int)
    # summarize the loaded data
    n_chars = len(raw_text)
    n_vocab = len(char_list)
    print("-> Total Characters: ", n_chars)
    print("-> Total Vocab: ", n_vocab)
    # prepare the dataset of input to output pairs encoded as integers
    seq_length = 100
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
    	seq_in = raw_text[i:i + seq_length]
    	seq_out = raw_text[i + seq_length]
    	dataX.append([char_to_int[char] for char in seq_in])
    	dataY.append(char_to_int[seq_out])



    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)
    # reshape X to be [samples, time steps, features]
    X = np.reshape(dataX, (n_patterns, seq_length, 1))
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)

    print('- Input:',X[0,:,:].shape)
    print('- Output:',y[0].shape)
    # X = np.squeeze(X)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.1,
                                                        random_state=42)

    # load_trained = True
    load_trained = False

    hidden_size = 256
    # for the feed forward layers after att
    hidden_att_size = 256
    # attention size
    att_size = 256
    # old
    rnn_size = 300
    batch_size = 64
    # if we want to change batch size during training
    # dynamic_batch = True
    dynamic_batch = False
    # bg_train = BalancedBatchGenerator(X_train, y_train, batch_size=batch_size, random_state=42)
    # bg_test = BalancedBatchGenerator(X_test, y_test, batch_size=batch_size, random_state=42)

    n_epochs = 40
    att_act = 'softmax'
    # dropout rate
    do_rate = 0.35
    # regularization
    kernel_regularizer = keras.regularizers.l2(0.0001)

    # opt = keras.optimizers.Adam()
    opt = keras.optimizers.Adadelta()

    sen_len = seq_length
    emb_len = n_vocab

    # which iteration of models to load
    current_it = 11
    next_it = 11
    # full_path = 'models/char_{}.h5'.format(current_it)
    full_path = 'models/char_9_epoch_6.h5'
    full_new_path = 'models/char_{}.h5'.format(next_it)



    if load_trained:
        print('-> Loading att model')
        with CustomObjectScope({'SeqSelfAttention': SeqSelfAttention,
                                'MultiHead': MultiHead,
                                'root_mean_squared_error': root_mean_squared_error}):
            model = keras.models.load_model(full_path)
    else:
        # define inputs
        inputs = Input(shape=(sen_len,1))
        # x = Dropout(do_rate)(inputs)
        # x = BatchNormalization()(inputs)

        x = GRU(units=rnn_size,
                # activity_regularizer=kernel_regularizer,
                # kernel_regularizer=kernel_regularizer,
                return_sequences=True)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(do_rate)(x)
        # x = GRU(units=hidden_att_size,
        #         activity_regularizer=kernel_regularizer,
        #         kernel_regularizer=kernel_regularizer,
        #         return_sequences=True)(x)
        # x = SeqSelfAttention(attention_activation=att_act,
        #                      units=att_size)(x)
        # x = Dropout(do_rate)(x)

        x = GRU(units=rnn_size)(x)
        x = Dropout(do_rate)(x)

        # tried really hard getting attention to work ¯\_(ツ)_/¯
        # x = SeqSelfAttention(attention_activation=att_act,
        #                      units=att_size)(x)

        # x = Flatten()(x)
        predictions = Dense(emb_len,
                            # activity_regularizer=kernel_regularizer,
                            # kernel_regularizer=kernel_regularizer,
                            activation='softmax')(x)

        model = Model(inputs=inputs, outputs=predictions)



        model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'categorical_accuracy'],
        )
    print(model.summary())


    for i in range(n_epochs):
        full_new_path = 'models/char_{}_epoch_{}.h5'.format(next_it, i)
        # set batch size dynamically
        if dynamic_batch:
            if i == 1:
                batch_size = 128
            elif i == 2:
                batch_size = 64
            else:
                batch_size = 32

        model.fit(X_train,
                  y_train,
                  batch_size = batch_size,
                  validation_data=(X_test, y_test))
        # fit using the batch generators
        # model.fit_generator(bg_train,
        #                     validation_data=bg_test,
        #                     use_multiprocessing=True,
        #                     workers=4,
        #                     epochs=1)
                            # validation_data=(oba_in_tests, oba_out_test))
        model.save(full_new_path)

if __name__ == '__main__':
    main()

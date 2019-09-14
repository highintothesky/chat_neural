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

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def main():
    # load test data
    filename = "data/lines_chars.txt"
    raw_text = open(filename, 'r', encoding='utf-8').read()
    raw_text = raw_text.lower()
    raw_text = re.sub('\n', " ", raw_text)

    # create mapping of unique chars to integers
    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    # summarize the loaded data
    n_chars = len(raw_text)
    n_vocab = len(chars)
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

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    # load_trained = True
    load_trained = False
    hidden_size = 256
    # for the feed forward layers after att
    hidden_att_size = 256
    # attention size
    att_size = 256
    # old
    rnn_size = 120
    batch_size = 128
    n_epochs = 2
    att_act = 'softmax'
    # dropout rate
    do_rate = 0.1
    # opt = keras.optimizers.Adam()
    opt = keras.optimizers.Adadelta()

    sen_len = seq_length
    emb_len = n_vocab

    # which iteration of models to load
    current_it = 1
    next_it = 1
    full_path = 'models/char_{}.h5'.format(current_it)
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
        x = BatchNormalization()(inputs)

        x = GRU(units=hidden_att_size,
                activation='selu',
                dropout=do_rate,
                recurrent_dropout=do_rate,
                return_sequences=True,
                # stateful=True,
                # return_state=True,
                recurrent_activation='hard_sigmoid',
                recurrent_initializer='glorot_uniform')(x)

        # x = GRU(units=hidden_att_size,
        #         activation='selu',
        #         dropout=do_rate,
        #         recurrent_dropout=do_rate,
        #         return_sequences=True,
        #         # stateful=True,
        #         # return_state=True,
        #         recurrent_activation='hard_sigmoid',
        #         recurrent_initializer='glorot_uniform')(x)

        x = SeqSelfAttention(attention_activation=att_act, units=att_size)(x)
        # x = SeqSelfAttention(attention_activation=att_act, units=att_size)(x)

        # dense = Dense(units=hidden_att_size,
        #               activation='selu')(x)
        # dense = Dropout(do_rate)(dense)
        # x = Add()([x, dense])
        # x = BatchNormalization()(x)
        # x = Flatten()(x)
        # x = SeqSelfAttention(attention_activation=att_act, units=att_size)(x)
        # dense = Dense(units=hidden_att_size,
        #               activation='selu')(x)
        # dense = Dropout(do_rate)(dense)
        # x = Add()([x, dense])
        # x = BatchNormalization()(x)
        #
        # x = SeqSelfAttention(attention_activation=att_act, units=att_size)(x)
        # dense = Dense(units=hidden_att_size,
        #               activation='selu')(x)
        # dense = Dropout(do_rate)(dense)
        # x = Add()([x, dense])

        x = GlobalMaxPool1D()(x)

        # x = SeqSelfAttention(attention_activation=att_act,
        #                      units=att_size)(x)

        x = BatchNormalization()(x)
        dense1 = Dense(units=hidden_size,
                       activation='selu')(x)
        dense1 = Dropout(do_rate)(dense1)
        dense1 = Dense(units=hidden_size,
                       activation='selu')(dense1)
        dense1 = Dropout(do_rate)(dense1)
        predictions = Dense(emb_len, activation='sigmoid')(dense1)

        model = Model(inputs=inputs, outputs=predictions)
        print(model.summary())
        # model.add(keras.layers.TimeDistributed(SeqSelfAttention(attention_activation='sigmoid'),
        #                                        input_shape=(sen_len, emb_len)))
        # model.add(keras.layers.GlobalMaxPool2D())
        # model.add(MultiHead(keras.layers.LSTM(units=32), layer_num=5, name='Multi-LSTMs'))
        # model.add(keras.layers.Bidirectional(keras.layers.GRU(units=hidden_size,
        #                           dropout=0.25,
        #                           recurrent_dropout=0.1,
        #                           return_sequences=True)))
        # model.add(MultiHead(GRU(units=70, return_sequences=True, dropout=0.25,recurrent_dropout=0.1),
        #                     layer_num=4, name='Multi-GRUs'))
        # model.add(MultiHead([SeqSelfAttention(attention_activation='sigmoid')]))
        # model.add(keras.layers.Bidirectional(keras.layers.GRU(units=rnn_size,
        #                   dropout=0.25,
        #                   recurrent_dropout=0.1,
        #                   return_sequences=True)))


    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'categorical_accuracy'],
    )

    # fit using the batch generators
    # model.fit_generator(bg,
    #                     validation_data=bg_test,
    #                     use_multiprocessing=True,
    #                     workers=6,
    #                     epochs=n_epochs)
    #                     # validation_data=(oba_in_tests, oba_out_test))
    model.fit(X_train,
              y_train,
              batch_size = batch_size,
              epochs=n_epochs,
              validation_data=(X_test, y_test))
    model.save(full_new_path)

if __name__ == '__main__':
    main()

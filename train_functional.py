# train using https://pypi.org/project/keras-self-attention/
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
import h5py
import numpy as np
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
    h5f = h5py.File('data/processed_bpe_test.h5', 'r')
    # restrict since my RAM isn't infinite
    # TODO: do batch loading from hdf
    # max_ind = int(len(h5f['input'])*0.1)
    # print('-> max index:', max_ind)
    oba_in_tests = h5f['input'][:]
    oba_out_test = np.squeeze(h5f['output'][:])
    print('-> output shape:', oba_out_test.shape)



    print(oba_in_tests.shape)
    # X_train, X_test, y_train, y_test = train_test_split(oba_in_tests,
    #                                                     oba_out_test,
    #                                                     test_size=0.8,
    #                                                     random_state=42)

    load_trained = True
    # load_trained = False

    hidden_size = 512
    # for the feed forward layers after att
    hidden_att_size = 512
    # attention size
    att_size = 512
    # old
    rnn_size = 120
    batch_size = 128
    n_epochs = 10
    att_act = 'softmax'

    # dropout rate
    do_rate = 0.1

    # opt = keras.optimizers.Adam()
    opt = keras.optimizers.Adadelta()
    sen_len = oba_in_tests.shape[1]
    emb_len = oba_in_tests.shape[2]
    # which iteration of models to load
    current_it = 9
    next_it = 9
    full_path = 'models/func_{}.h5'.format(current_it)
    full_new_path = 'models/func_{}.h5'.format(next_it)

    # create a list of h5 for batchgenerator
    h5_list = [h5py.File('data/processed_bpe_1.h5', 'r'),
               h5py.File('data/processed_bpe_2.h5', 'r'),
               h5py.File('data/processed_bpe_3.h5', 'r')]

    bg = BatchGenerator(h5_list, batch_size)
    bg_test = BatchGenerator([h5f], batch_size)

    if load_trained:
        print('-> Loading att model')
        with CustomObjectScope({'SeqSelfAttention': SeqSelfAttention,
                                'MultiHead': MultiHead,
                                'root_mean_squared_error': root_mean_squared_error}):
            model = keras.models.load_model(full_path)
                                        # custom_objects=SeqSelfAttention.get_custom_objects())
        # model = keras.models.load_model(full_path,
        #                                 custom_objects=SeqSelfAttention.get_custom_objects())
    else:
        # define inputs
        inputs = Input(shape=(sen_len, emb_len))
        x = BatchNormalization()(inputs)


        # x = MultiHead(SeqSelfAttention(attention_activation=att_act, units=att_size))(x)
        # x = TimeDistributed(Dense(units=hidden_att_size,activation='selu'))(x)
        # x = Concatenate()(x)

        x = GRU(units=hidden_att_size,
                activation='selu',
                dropout=do_rate,
                recurrent_dropout=do_rate,
                return_sequences=False,
                # stateful=True,
                # return_state=True,
                recurrent_activation='hard_sigmoid',
                recurrent_initializer='glorot_uniform')(x)

        # x = SeqSelfAttention(attention_activation=att_act, units=att_size)(bn)
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

        # x = GlobalMaxPool2D()(x)

        # x = SeqSelfAttention(attention_activation=att_act,
        #                      units=att_size)(x)

        x = BatchNormalization()(x)
        dense1 = Dense(units=hidden_size,
                       activation='selu')(x)
        dense1 = Dropout(do_rate)(dense1)
        dense1 = Dense(units=hidden_size,
                       activation='selu')(dense1)
        dense1 = Dropout(do_rate)(dense1)
        predictions = Dense(emb_len, activation='linear')(dense1)

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
        loss='mae',
        metrics=['mse', 'mae', root_mean_squared_error],
    )

    # fit using the batch generators
    model.fit_generator(bg,
                        validation_data=bg_test,
                        use_multiprocessing=True,
                        workers=6,
                        epochs=n_epochs)
                        # validation_data=(oba_in_tests, oba_out_test))
    # model.fit(X_train,
    #           y_train,
    #           batch_size = batch_size,
    #           epochs=n_epochs,
    #           validation_data=(X_test, y_test))
    model.save(full_new_path)

if __name__ == '__main__':
    main()

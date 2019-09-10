# train using https://pypi.org/project/keras-self-attention/
import keras
import h5py
import numpy as np
from keras_self_attention import SeqSelfAttention
from keras_multi_head import MultiHead
from keras.utils import CustomObjectScope
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Dropout, GRU, GlobalMaxPool1D, GlobalMaxPool2D, BatchNormalization, Add, Flatten, TimeDistributed

def main():
    # load data
    h5f = h5py.File('data/processed_bpe.h5', 'r')
    max_ind = int(len(h5f['input'])*0.5)
    print('-> max index:', max_ind)
    oba_inputs = h5f['input'][:max_ind]
    oba_outputs = np.squeeze(h5f['output'][:max_ind])
    print('-> output shape:', oba_outputs.shape)
    # load_trained = True
    load_trained = True
    # print(oba_inputs[0,:,:])
    # print(oba_outputs[0,:,:])

    print(oba_inputs.shape)
    X_train, X_test, y_train, y_test = train_test_split(oba_inputs,
                                                        oba_outputs,
                                                        test_size=0.25,
                                                        random_state=42)


    hidden_size = 150
    rnn_size = 100
    batch_size = 64
    n_epochs = 3
    sen_len = oba_inputs.shape[1]
    emb_len = oba_inputs.shape[2]
    # which iteration of models to load
    current_it = 11
    next_it = 11
    full_path = 'models/att{}_full.h5'.format(current_it)
    full_new_path = 'models/att{}_full.h5'.format(next_it)

    if load_trained:
        print('-> Loading att model')
        with CustomObjectScope({'SeqSelfAttention': SeqSelfAttention,
                                'MultiHead': MultiHead}):
            model = keras.models.load_model(full_path)
                                        # custom_objects=SeqSelfAttention.get_custom_objects())
        # model = keras.models.load_model(full_path,
        #                                 custom_objects=SeqSelfAttention.get_custom_objects())
    else:
        model = keras.models.Sequential()
        # model.add(MultiHead(GRU(units=70, return_sequences=True, dropout=0.25,recurrent_dropout=0.1),
        #                     layer_num=4, name='Multi-GRUs'))
        # model.add(MultiHead([SeqSelfAttention(attention_activation='sigmoid')]))
        model.add(keras.layers.Bidirectional(keras.layers.GRU(units=rnn_size,
                          dropout=0.25,
                          recurrent_dropout=0.1,
                          return_sequences=True)))
        # model.add(SeqSelfAttention(attention_activation='sigmoid'))
        model.add(keras.layers.Bidirectional(keras.layers.GRU(units=rnn_size,
                          dropout=0.25,
                          recurrent_dropout=0.1,
                          return_sequences=True)))
        # model.add(SeqSelfAttention(attention_activation='sigmoid'))
        # model.add(keras.layers.Bidirectional(keras.layers.GRU(units=rnn_size,
        #                   dropout=0.25,
        #                   recurrent_dropout=0.1,
        #                   return_sequences=False)))
        model.add(SeqSelfAttention(attention_activation='sigmoid'))

        model.add(BatchNormalization())
        # model.add(Flatten())
        model.add(keras.layers.GlobalMaxPool1D())

        model.add(keras.layers.Dense(units=hidden_size, activation='selu'))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(units=hidden_size, activation='selu'))
        model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.Dense(emb_len, activation='linear'))


        # model.add(TimeDistributed(SeqSelfAttention(attention_activation='sigmoid')))
        # model.add(keras.layers.Dense(units=hidden_size, activation='selu'))
        # model.add(keras.layers.Dropout(0.25))
        # model.add(keras.layers.GlobalMaxPool1D())

        # model.add(MultiHead(keras.layers.LSTM(units=32), layer_num=5, name='Multi-LSTMs'))
        # model.add(keras.layers.TimeDistributed(SeqSelfAttention(attention_activation='sigmoid'),
        #                                        input_shape=(sen_len, emb_len)))
        # model.add(keras.layers.GlobalMaxPool2D())
        # model.add(MultiHead(keras.layers.LSTM(units=32), layer_num=5, name='Multi-LSTMs'))
        # model.add(keras.layers.Bidirectional(keras.layers.GRU(units=hidden_size,
        #                           dropout=0.25,
        #                           recurrent_dropout=0.1,
        #                           return_sequences=True)))
        # model.add(SeqSelfAttention(attention_activation='sigmoid'))
        # model.add(keras.layers.GlobalMaxPool1D())
        # model.add(keras.layers.Dense(units=hidden_size, activation='selu'))
        # model.add(keras.layers.Dropout(0.25))

        # model.add(keras.layers.LSTM(hidden_size, input_shape=(sen_len, emb_len)))


        # we might want to recompile
        model.compile(
            optimizer='Adam',
            loss='mse',
            metrics=['mse', 'mae'],
        )
        # model.build((X_train, y_train))
        # print(model.summary())

    model.fit(X_train,
              y_train,
              batch_size = batch_size,
              epochs=n_epochs,
              validation_data=(X_test, y_test))
    model.save(full_new_path)

if __name__ == '__main__':
    main()

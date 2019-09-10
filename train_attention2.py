# train using https://pypi.org/project/keras-self-attention/
import keras
import h5py
from keras_self_attention import SeqSelfAttention
from sklearn.model_selection import train_test_split


def main():
    # load data
    h5f = h5py.File('data/processed1.h5', 'r')
    max_ind = int(len(h5f['utterances'])*1.)
    oba_utt = h5f['utterances'][:max_ind]
    oba_resp = h5f['responses'][:max_ind]
    load_trained = False

    print(oba_utt.shape)
    X_train, X_test, y_train, y_test = train_test_split(oba_utt,
                                                        oba_resp,
                                                        test_size=0.25,
                                                        random_state=42)


    hidden_size = 200
    batch_size = 256
    n_epochs = 10
    sen_len = oba_utt.shape[1]
    emb_len = oba_utt.shape[2]
    # which iteration of models to load
    current_it = 1
    next_it = 1
    full_path = 'models/att{}_full.h5'.format(current_it)
    full_new_path = 'models/att{}_full.h5'.format(next_it)

    if load_trained:
        print('-> Loading att model')
        model = keras.models.load_model(full_path,
                                        custom_objects=SeqSelfAttention.get_custom_objects())
    else:
        model = keras.models.Sequential()
        model.add(keras.layers.Bidirectional(keras.layers.GRU(units=hidden_size,
                                                               return_sequences=True)))
        model.add(SeqSelfAttention(attention_activation='sigmoid'))
        model.add(keras.layers.Dense(units=hidden_size, activation='relu'))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(units=emb_len))
        # we might want to recompile
        model.compile(
            optimizer='Adam',
            loss='mse',
            metrics=['mse', 'mae'],
        )


    model.fit(X_train,
              y_train,
              batch_size = batch_size,
              epochs=n_epochs,
              validation_data=(X_test, y_test))
    model.save(full_new_path)

if __name__ == '__main__':
    main()

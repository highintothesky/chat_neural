# train using https://pypi.org/project/keras-self-attention/
import keras
import h5py
from keras_self_attention import SeqSelfAttention


def main():
    # load data
    h5f = h5py.File('data/processed1.h5', 'r')
    oba_utt = h5f['utterances'][:]
    oba_resp = h5f['responses'][:]
    load_trained = False

    print(oba_utt.shape)


    hidden_size = 100
    batch_size = 512
    n_epochs = 10
    sen_len = oba_utt.shape[1]
    emb_len = oba_utt.shape[2]
    # which iteration of models to load
    current_it = 4
    next_it = 4
    full_path = 'models/att{}_full.h5'.format(current_it)
    full_new_path = 'models/att{}_full.h5'.format(next_it)

    if load_trained:
        print('-> Loading att model')
        # model = load_model(full_path,
        #                        custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform,
        #                                        "AttentionLayer": k_att.AttentionLayer})

    else:
        model = keras.models.Sequential()
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=128,
                                                               return_sequences=True)))
        model.add(SeqSelfAttention(attention_activation='sigmoid'))
        model.add(keras.layers.Dense(units=emb_len))
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mse', 'mae'],
        )


    model.fit(oba_utt,
              oba_resp,
              batch_size = batch_size,
              epochs=n_epochs)
    model.save(full_new_path)

if __name__ == '__main__':
    main()

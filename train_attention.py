# train a keras attention model
import h5py
import numpy as np
import tensorflow as tf
import layers.attention as k_att
from tensorflow.keras.layers import Input, GRU, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
# from keras.layers import Input, GRU, Dense, Concatenate, TimeDistributed
# from keras.models import Model
from layers.attention import AttentionLayer
from process_movie_lines import sentences_to_matrix


def define_nmt(hidden_size, batch_size, en_timesteps, en_vsize, fr_timesteps, fr_vsize):
    """ Defining a NMT model """

    # Define an input sequence and process it.
    if batch_size:
        encoder_inputs = Input(batch_shape=(batch_size, en_timesteps, en_vsize), name='encoder_inputs')
        decoder_inputs = Input(batch_shape=(batch_size, fr_timesteps - 1, fr_vsize), name='decoder_inputs')
    else:
        encoder_inputs = Input(shape=(en_timesteps, en_vsize), name='encoder_inputs')
        decoder_inputs = Input(shape=(fr_timesteps - 1, fr_vsize), name='decoder_inputs')

    # Encoder GRU
    encoder_gru = GRU(hidden_size, return_sequences=True, return_state=True, name='encoder_gru')
    encoder_out, encoder_state = encoder_gru(encoder_inputs)

    # Set up the decoder GRU, using `encoder_states` as initial state.
    decoder_gru = GRU(hidden_size, return_sequences=True, return_state=True, name='decoder_gru')
    decoder_out, decoder_state = decoder_gru(decoder_inputs, initial_state=encoder_state)

    # Attention layer
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_out, decoder_out])

    # Concat attention input and decoder GRU output
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])

    # Dense layer
    dense = Dense(fr_vsize, activation='linear', name='softmax_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    decoder_pred = dense_time(decoder_concat_input)

    # Full model
    full_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
    # full_model.compile(optimizer='adam', loss='categorical_crossentropy')
    full_model.compile(optimizer='adam', loss='mse')

    full_model.summary()

    """ Inference model """
    batch_size = 1

    """ Encoder (Inference) model """
    encoder_inf_inputs = Input(batch_shape=(batch_size, en_timesteps, en_vsize), name='encoder_inf_inputs')
    encoder_inf_out, encoder_inf_state = encoder_gru(encoder_inf_inputs)
    encoder_model = Model(inputs=encoder_inf_inputs, outputs=[encoder_inf_out, encoder_inf_state])

    """ Decoder (Inference) model """
    decoder_inf_inputs = Input(batch_shape=(batch_size, 1, fr_vsize), name='decoder_word_inputs')
    encoder_inf_states = Input(batch_shape=(batch_size, en_timesteps, hidden_size), name='encoder_inf_states')
    decoder_init_state = Input(batch_shape=(batch_size, hidden_size), name='decoder_init')

    decoder_inf_out, decoder_inf_state = decoder_gru(decoder_inf_inputs, initial_state=decoder_init_state)
    attn_inf_out, attn_inf_states = attn_layer([encoder_inf_states, decoder_inf_out])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_inf_out, attn_inf_out])
    decoder_inf_pred = TimeDistributed(dense)(decoder_inf_concat)
    decoder_model = Model(inputs=[encoder_inf_states, decoder_init_state, decoder_inf_inputs],
                          outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_state])

    return full_model, encoder_model, decoder_model

def infer_nmt(encoder_model, decoder_model, test_en_seq, emb_len, ft_model):
    """
    Infer logic
    :param encoder_model: keras.Model
    :param decoder_model: keras.Model
    :param test_en_seq: sequence of word ids
    :param en_vsize: int
    :param fr_vsize: int
    :return:
    """

    # test_fr_seq = sents2sequences(fr_tokenizer, ['sos'], emb_len)
    # test_en_onehot_seq = to_categorical(test_en_seq, num_classes=emb_len)
    # test_fr_onehot_seq = np.expand_dims(to_categorical(test_fr_seq, num_classes=emb_len), 1)
    test_en_onehot_seq = test_en_seq
    test_fr_seq = np.array(ft_model['sos'])
    test_fr_onehot_seq = np.expand_dims(test_fr_seq, 0)
    test_fr_onehot_seq = np.expand_dims(test_fr_onehot_seq, 0)

    enc_outs, enc_last_state = encoder_model.predict(test_en_onehot_seq)
    dec_state = enc_last_state
    attention_weights = []
    out_list = [test_fr_seq]
    fr_text = ''
    for i in range(10):
        # print('-> current test_ft out shape:')
        # print(test_fr_onehot_seq.shape)

        dec_out, attention, dec_state = decoder_model.predict([enc_outs, dec_state, test_fr_onehot_seq])
        dec_ind = np.argmax(dec_out, axis=-1)[0, 0]
        # print('-> decoder index:', dec_ind)
        # print('-> decoder out:')
        # print(dec_out.shape)
        # print(dec_out)

        if dec_ind == 0:
            print('-> breaking at 0')
            break
        out_list.append(dec_out)
        # test_fr_seq = np.array(out_list)
        # test_fr_onehot_seq = test_fr_seq
        arr = np.zeros((1, i+1, emb_len))
        for j in range(i + 1):
            arr[0,j,:] = out_list[j]
        # print('-> arr shape')
        # print(arr.shape)
        test_fr_onehot_seq = dec_out
        # print('-> test_fr_seq shape:')
        # print(test_fr_seq.shape)
        # print('-> WORD:')
        word = ft_model.most_similar(positive=[dec_out[0,0,:]])[0][0]
        # print(word)
        # test_fr_seq = sents2sequences(fr_tokenizer, [fr_index2word[dec_ind]], emb_len)
        # test_fr_onehot_seq = np.expand_dims(test_fr_seq, 0)

        attention_weights.append((dec_ind, attention))
        fr_text += word + ' '

    # print(out_list)
    print(attention_weights)
    return fr_text, attention_weights

def train(full_model, en_seq, fr_seq, batch_size, n_epochs=10):
    """ Training the model """

    for ep in range(n_epochs):
        losses = []
        for bi in range(0, en_seq.shape[0] - batch_size, batch_size):

            en_onehot_seq = en_seq[bi:bi + batch_size, :, :]
            fr_onehot_seq = fr_seq[bi:bi + batch_size, :, :]

            full_model.train_on_batch([en_onehot_seq, fr_onehot_seq[:, :-1, :]], fr_onehot_seq[:, 1:, :])

            l = full_model.evaluate([en_onehot_seq, fr_onehot_seq[:, :-1, :]], fr_onehot_seq[:, 1:, :],
                                    batch_size=batch_size, verbose=0)

            losses.append(l)
        if (ep + 1) % 1 == 0:
            print("Loss in epoch {}: {}".format(ep + 1, np.mean(losses)))

def main():
    # load data
    h5f = h5py.File('data/processed1.h5', 'r')
    oba_utt = h5f['utterances'][:]
    oba_resp = h5f['responses'][:]
    load_trained = True

    print(oba_utt.shape)


    hidden_size = 100
    batch_size = 512
    n_epochs = 10
    sen_len = oba_utt.shape[1]
    emb_len = oba_utt.shape[2]
    # which iteration of models to load
    current_it = 2
    next_it = 3
    full_path = 'models/att{}_full.h5'.format(current_it)
    full_new_path = 'models/att{}_full.h5'.format(next_it)
    enc_path = 'models/att{}_enc.h5'.format(current_it)
    enc_new_path = 'models/att{}_enc.h5'.format(next_it)
    dec_path = 'models/att{}_dec.h5'.format(current_it)
    dec_new_path = 'models/att{}_dec.h5'.format(next_it)

    if load_trained:
        print('-> Loading att model')
        full_model = load_model(full_path,
                               custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform,
                                               "AttentionLayer": k_att.AttentionLayer})
        # print('-> Loading enc model')
        # infer_enc_model = load_model(enc_path,
        #                        custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform,
        #                                        "AttentionLayer": k_att.AttentionLayer})
        # print('-> Loading dec model')
        #
        # infer_dec_model = load_model(dec_path,
        #                        custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform,
        #                                        "AttentionLayer": k_att.AttentionLayer})
    else:
        full_model, infer_enc_model, infer_dec_model = define_nmt(
            hidden_size=hidden_size, batch_size=None,
            # hidden_size=hidden_size,
            en_timesteps=sen_len, fr_timesteps=sen_len,
            en_vsize=emb_len, fr_vsize=emb_len)
    train(full_model, oba_utt, oba_resp, batch_size, n_epochs)

    full_model.save(full_new_path)
    # infer_enc_model.save(enc_new_path)
    # infer_dec_model.save(dec_new_path)

if __name__ == '__main__':
    main()

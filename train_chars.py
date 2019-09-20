# train using https://pypi.org/project/keras-self-attention/
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import re
# import tensorflow as tf
# from tensorflow import keras
# import h5py
# import numpy as np

import os
import re
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.utils import get_file, to_categorical, CustomObjectScope
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy
# from tensorflow.keras import Model
# from tensorflow.keras.layers import Layer, Lambda, Input, Embedding, Bidirectional, Flatten, Dense, GRU, Dropout, BatchNormalization
# from tensorflow.compat.v1.keras.layers import CuDNNLSTM
# # from https://github.com/ongunuzaymacar/attention-mechanisms
# from attention.layers import Attention, SelfAttention
from attention_model import AttentionModel

# def root_mean_squared_error(y_true, y_pred):
#     return K.sqrt(K.mean(K.square(y_pred - y_true)))

def perplexity(labels, logits):
    """Calculates perplexity metric = 2^(entropy) or e^(entropy)"""
    return pow(2, categorical_crossentropy(y_true=labels, y_pred=logits))

def main():
    # load test data
    filename = "data/mixed.txt"
    raw_text = open(filename, 'r', encoding='utf-8').read()
    print('-> Raw text length:', len(raw_text))
    raw_text = raw_text.lower()[:700000]
    raw_text = re.sub('\n', " ", raw_text)

    # create mapping of unique chars to integers
    chars = sorted(list(set(raw_text)))
    # print('-> chars:', chars)
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    print('-> int to char:', int_to_char)
    print('-> char to int:', char_to_int)
    # print(char_to_int)
    # char_to_int = {' ': 0, '!': 1, '%': 2, '&': 3, "'": 4, ',': 5, '-': 6, '.': 7, '/': 8, '0': 9, '1': 10, '2': 11, '3': 12, '4': 13, '5': 14, '6': 15, '7': 16, '8': 17, '9': 18, ':': 19, ';': 20, '<': 21, '>': 22, '?': 23, 'a': 24, 'b': 25, 'c': 26, 'd': 27, 'e': 28, 'f': 29, 'g': 30, 'h': 31, 'i': 32, 'j': 33, 'k': 34, 'l': 35, 'm': 36, 'n': 37, 'o': 38, 'p': 39, 'q': 40, 'r': 41, 's': 42, 't': 43, 'u': 44, 'v': 45, 'w': 46, 'x': 47, 'y': 48, 'z': 49, '~': 50, '—': 51}
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
    sentences = []
    next_chars = []
    for i in range(0, n_chars - seq_length, 1):
        sentences.append(raw_text[i:i+seq_length])
        next_chars.append(raw_text[i+seq_length])

    n_patterns = len(sentences)
    print("Total Patterns: ", n_patterns)

    X = np.zeros((n_patterns, seq_length, n_vocab), dtype=np.bool)
    y = np.zeros((n_patterns, n_vocab), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_to_int[char]] = 1
        y[i, char_to_int[next_chars[i]]] = 1


    print('- Input:',X[0,:,:].shape)
    print('- Output:',y[0].shape)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    batch_size = 100

    # if we want to change batch size during training
    # dynamic_batch = True
    dynamic_batch = False

    n_epochs = 40
    # opt = keras.optimizers.Adam()
    # opt = keras.optimizers.Adadelta()
    opt = keras.optimizers.RMSprop(lr=0.001)

    sen_len = seq_length
    emb_len = n_vocab

    # which iteration of models to load
    # next_it = 2
    #     # encoder_output, attention_weights = SelfAttention(size=50,
    #     #                                                   num_hops=16,
    #     #                                                   use_penalization=False)(x)

    checkpoint_path = "models/char_att3/"

    # Create checkpoint callback
    cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                  monitor='val_accuracy',
                                                  verbose=1,
                                                  save_weights_only=True,
                                                  mode='max',
                                                  save_best_only=True)

    load_model = False
    # load_model = True

    am = AttentionModel(checkpoint_path = checkpoint_path,
                        rnn_size = 512,
                        rnn_style = 'CuDNNLSTM',
                        dropout_rate = 0.3,
                        load_model = load_model)
    # am.build_model()
    am.model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy', perplexity, categorical_accuracy],
    )
    # print(am.model.summary())
    am.save_config()

    am.model.fit(X_train,
                 y_train,
                 batch_size = batch_size,
                 validation_data=(X_test, y_test),
                 callbacks = [cp_callback],
                 epochs=n_epochs)
    # for i in range(n_epochs):
    #     full_new_path = 'models/char_att_{}_epoch_{}.h5'.format(next_it, i)
    #     # set batch size dynamically
    #     if dynamic_batch:
    #         if i == 1:
    #             batch_size = 128
    #         elif i == 2:
    #             batch_size = 64
    #         else:
    #             batch_size = 32

        # fit using the batch generators
        # model.fit_generator(bg_train,
        #                     validation_data=bg_test,
        #                     use_multiprocessing=True,
        #                     workers=4,
        #                     epochs=40)
                            # validation_data=(oba_in_tests, oba_out_test))
        # print('-> Saving to', full_new_path)
        # with CustomObjectScope({'SelfAttention': SelfAttention,
        #                         'Attention': Attention,
        #                         'perplexity': perplexity}):
        # model.save(full_new_path)

if __name__ == '__main__':
    main()

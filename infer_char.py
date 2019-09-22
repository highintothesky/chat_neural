# test the trained model v3
# import re
import numpy as np
# import keras
import random
import sys
import io
# import keras_self_attention
from keras_multi_head import MultiHead
# from keras.utils import CustomObjectScope
import re
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import get_file, to_categorical, CustomObjectScope
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Lambda, Input, Embedding, Bidirectional, Flatten, Dense, GRU
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from attention.layers import Attention, SelfAttention
from attention_model import AttentionModel

def perplexity(labels, logits):
    """Calculates perplexity metric = 2^(entropy) or e^(entropy)"""
    return pow(2, categorical_crossentropy(y_true=labels, y_pred=logits))

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

if __name__ == '__main__':
    print('-> Loading att model')

    checkpoint_path = "models/char_att6/"
    model = AttentionModel(checkpoint_path = checkpoint_path,
                            load_model = True).model

    filename = "data/mixed.txt"
    raw_text = open(filename, 'r', encoding='utf-8').read()
    raw_text = raw_text.lower()
    raw_text = re.sub('\n', " ", raw_text)[:10000]

    # create mapping of unique chars to integers
    chars = sorted(list(set(raw_text)))
    # char_to_int = dict((c, i) for i, c in enumerate(chars))
    # int_to_char = dict((i, c) for i, c in enumerate(chars))
    int_to_char = {0: ' ', 1: '!', 2: '"', 3: "'", 4: ',', 5: '-', 6: '.', 7: '/', 8: '0', 9: '1', 10: '2', 11: '3', 12: '4', 13: '5', 14: '6', 15: '7', 16: '8', 17: '9', 18: ':', 19: ';', 20: '<', 21: '>', 22: '?', 23: '@', 24: 'a', 25: 'b', 26: 'c', 27: 'd', 28: 'e', 29: 'f', 30: 'g', 31: 'h', 32: 'i', 33: 'j', 34: 'k', 35: 'l', 36: 'm', 37: 'n', 38: 'o', 39: 'p', 40: 'q', 41: 'r', 42: 's', 43: 't', 44: 'u', 45: 'v', 46: 'w', 47: 'x', 48: 'y', 49: 'z', 50: '~'}
    char_to_int = {' ': 0, '!': 1, '"': 2, "'": 3, ',': 4, '-': 5, '.': 6, '/': 7, '0': 8, '1': 9, '2': 10, '3': 11, '4': 12, '5': 13, '6': 14, '7': 15, '8': 16, '9': 17, ':': 18, ';': 19, '<': 20, '>': 21, '?': 22, '@': 23, 'a': 24, 'b': 25, 'c': 26, 'd': 27, 'e': 28, 'f': 29, 'g': 30, 'h': 31, 'i': 32, 'j': 33, 'k': 34, 'l': 35, 'm': 36, 'n': 37, 'o': 38, 'p': 39, 'q': 40, 'r': 41, 's': 42, 't': 43, 'u': 44, 'v': 45, 'w': 46, 'x': 47, 'y': 48, 'z': 49, '~': 50}
    # print(char_to_int)
    # char_to_int = {' ': 0, '!': 1, '%': 2, '&': 3, "'": 4, ',': 5, '-': 6, '.': 7, '/': 8, '0': 9, '1': 10, '2': 11, '3': 12, '4': 13, '5': 14, '6': 15, '7': 16, '8': 17, '9': 18, ':': 19, ';': 20, '<': 21, '>': 22, '?': 23, 'a': 24, 'b': 25, 'c': 26, 'd': 27, 'e': 28, 'f': 29, 'g': 30, 'h': 31, 'i': 32, 'j': 33, 'k': 34, 'l': 35, 'm': 36, 'n': 37, 'o': 38, 'p': 39, 'q': 40, 'r': 41, 's': 42, 't': 43, 'u': 44, 'v': 45, 'w': 46, 'x': 47, 'y': 48, 'z': 49, '~': 50, '—': 51}
    char_list = list(char_to_int.keys())
    raw_text = ''.join([i for i in raw_text if i in char_list])

    # int_to_char = dict((i, c) for i, c in enumerate(chars))
    # int_to_char = {0: ' ', 1: '!', 2: '%', 3: '&', 4: "'", 5: ',', 6: '-', 7: '.', 8: '/', 9: '0', 10: '1', 11: '2', 12: '3', 13: '4', 14: '5', 15: '6', 16: '7', 17: '8', 18: '9', 19: ':', 20: ';', 21: '<', 22: '>', 23: '?', 24: 'a', 25: 'b', 26: 'c', 27: 'd', 28: 'e', 29: 'f', 30: 'g', 31: 'h', 32: 'i', 33: 'j', 34: 'k', 35: 'l', 36: 'm', 37: 'n', 38: 'o', 39: 'p', 40: 'q', 41: 'r', 42: 's', 43: 't', 44: 'u', 45: 'v', 46: 'w', 47: 'x', 48: 'y', 49: 'z', 50: '~', 51: '—'}

    # summarize the loaded data
    n_chars = len(raw_text)
    n_vocab = len(char_list)
    print("-> Total Characters: ", n_chars)
    print("-> Total Vocab: ", n_vocab)

    n_chars = len(raw_text)
    # prepare the dataset of input to output pairs encoded as integers
    seq_length = 100
    sentences = []
    next_chars = []
    for i in range(0, n_chars - seq_length, 1):
        sentences.append(raw_text[i:i+seq_length])
        next_chars.append(raw_text[i+seq_length])

    n_patterns = len(sentences)
    print("Total Patterns: ", n_patterns)
    print("n_vocab:", n_vocab)

    X = np.zeros((n_patterns, seq_length, n_vocab), dtype=np.bool)
    y = np.zeros((n_patterns, n_vocab), dtype=np.bool)

    print(X.shape)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_to_int[char]] = 1
        y[i, char_to_int[next_chars[i]]] = 1


    start_index = random.randint(0, len(raw_text) - seq_length - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = raw_text[start_index: start_index + seq_length]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, seq_length, n_vocab))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_to_int[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = int_to_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


    # # reshape X to be [samples, time steps, features]
    # X = np.reshape(dataX, (n_patterns, seq_length, 1))
    # # normalize
    # X = X / float(n_vocab)
    # # one hot encode the output variable
    # y = np_utils.to_categorical(dataY)
    #
    # for i in range(len(sentences)):
    #     # pick a random seed
    #     # start = numpy.random.randint(0, len(dataX)-1)
    #     start = np.random.randint(0, len(dataX)-1)
    #     pattern = dataX[start]
    #     res = ''
    #     print('pattern:', pattern)
    #     print( "Seed:")
    #     print( "\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    #     # generate characters
    #     for i in range(100):
    #         x = np.reshape(pattern, (1, len(pattern), 1))
    #         x = x / float(n_vocab)
    #         prediction = model.predict(x, verbose=0)
    #         index = np.argmax(prediction)
    #         # print('index pred:', index)
    #         result = int_to_char[index]
    #         seq_in = [int_to_char[value] for value in pattern]
    #         # sys.stdout.write(result)
    #         res += result
    #         pattern.append(index)
    #         pattern = pattern[1:len(pattern)]
    #
    #     # print('-> original sentence:', sentences[i])
    #
    #
    #
    #     print('-> output sentence:', res)
    #     print('--------------------------------')
        # print('-> max scoring for first pred:', sent_oba)
        # print('-> output sentence cos:', sentence_test_cos)

        # print(' ')
        # print('-> output second level:')
        # print(sentence2)

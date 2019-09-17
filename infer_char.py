# test the trained model v3
import re
import numpy as np
import keras
# import keras_self_attention
from keras_multi_head import MultiHead
from keras.utils import CustomObjectScope
from process_movie_lines3 import embed_single
from train_functional import root_mean_squared_error
from keras.models import load_model
from keras.utils import np_utils

from keras_self_attention import SeqSelfAttention
from gensim.models import FastText


if __name__ == '__main__':
    print('-> Loading att model')

    with CustomObjectScope({'SeqSelfAttention': SeqSelfAttention,
                            'MultiHead': MultiHead,
                            'root_mean_squared_error': root_mean_squared_error}):
        model = keras.models.load_model('models/char_10_epoch_4.h5')

    print(model.summary())
    sentences = ["hey, what's up?",
                 "kiss my ass",
                 "Are you really a robot? I can't believe it! now take this long string and do something with it",
                 "Gosh if only we could find Kat a boyfriend",
                 "you"]
    # embeddings = []
    # for idx, line in enumerate(sentences):
    #     line = line.translate({ord(i): None for i in """[]'\""""})
    #     line = re.sub('<u>', '', line)
    #     line = re.sub('</u>', '', line)
    #     line = re.sub('<U>', '', line)
    #     line = re.sub('</U>', '', line)
    #     line = re.sub('\`', " ", line)
    #     line = re.sub('\-\-', ' ', line)
    #     line = re.sub('\. \. \.', ' . ', line)
    #     line = re.sub('\.  \.  \.', ' . ', line)
    #     line = re.sub(' +', ' ', line)
    #     line = line.lower()
    #     sentences[idx] = line
        # embedded = embed_single(line, bpemb_en, seq_length)
        # print(embedded)
        # embeddings.append(embedded)
    # print('-> test mat shape:', embedded.shape)

    filename = "data/memes.txt"
    raw_text = open(filename, 'r', encoding='utf-8').read()
    raw_text = raw_text.lower()
    raw_text = re.sub('\n', " ", raw_text)

    # create mapping of unique chars to integers
    chars = sorted(list(set(raw_text)))
    # char_to_int = dict((c, i) for i, c in enumerate(chars))
    # int_to_char = dict((i, c) for i, c in enumerate(chars))
    char_to_int = {' ': 0, '!': 1, '%': 2, '&': 3, "'": 4, ',': 5, '-': 6, '.': 7, '/': 8, '0': 9, '1': 10, '2': 11, '3': 12, '4': 13, '5': 14, '6': 15, '7': 16, '8': 17, '9': 18, ':': 19, ';': 20, '<': 21, '>': 22, '?': 23, 'a': 24, 'b': 25, 'c': 26, 'd': 27, 'e': 28, 'f': 29, 'g': 30, 'h': 31, 'i': 32, 'j': 33, 'k': 34, 'l': 35, 'm': 36, 'n': 37, 'o': 38, 'p': 39, 'q': 40, 'r': 41, 's': 42, 't': 43, 'u': 44, 'v': 45, 'w': 46, 'x': 47, 'y': 48, 'z': 49, '~': 50, '—': 51}
    char_list = list(char_to_int.keys())
    raw_text = ''.join([i for i in raw_text if i in char_list])
    print(raw_text[0:300])
    # int_to_char = dict((i, c) for i, c in enumerate(chars))
    int_to_char = {0: ' ', 1: '!', 2: '%', 3: '&', 4: "'", 5: ',', 6: '-', 7: '.', 8: '/', 9: '0', 10: '1', 11: '2', 12: '3', 13: '4', 14: '5', 15: '6', 16: '7', 17: '8', 18: '9', 19: ':', 20: ';', 21: '<', 22: '>', 23: '?', 24: 'a', 25: 'b', 26: 'c', 27: 'd', 28: 'e', 29: 'f', 30: 'g', 31: 'h', 32: 'i', 33: 'j', 34: 'k', 35: 'l', 36: 'm', 37: 'n', 38: 'o', 39: 'p', 40: 'q', 41: 'r', 42: 's', 43: 't', 44: 'u', 45: 'v', 46: 'w', 47: 'x', 48: 'y', 49: 'z', 50: '~', 51: '—'}
    print('-> int to char:', int_to_char)
    # summarize the loaded data
    n_chars = len(raw_text)
    n_vocab = len(char_list)
    print("-> Total Characters: ", n_chars)
    print("-> Total Vocab: ", n_vocab)

    # we're not going to process all that
    # raw_text = ' '.join([line for line in sentences])
    n_chars = len(raw_text)
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

    # reshape X to be [samples, time steps, features]
    X = np.reshape(dataX, (n_patterns, seq_length, 1))
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)

    for i in range(len(sentences)):
        # pick a random seed
        # start = numpy.random.randint(0, len(dataX)-1)
        start = np.random.randint(0, len(dataX)-1)
        pattern = dataX[start]
        res = ''
        print('pattern:', pattern)
        print( "Seed:")
        print( "\"", ''.join([int_to_char[value] for value in pattern]), "\"")
        # generate characters
        for i in range(100):
            x = np.reshape(pattern, (1, len(pattern), 1))
            x = x / float(n_vocab)
            prediction = model.predict(x, verbose=0)
            index = np.argmax(prediction)
            # print('index pred:', index)
            result = int_to_char[index]
            seq_in = [int_to_char[value] for value in pattern]
            # sys.stdout.write(result)
            res += result
            pattern.append(index)
            pattern = pattern[1:len(pattern)]

        # print('-> original sentence:', sentences[i])



        print('-> output sentence:', res)
        print('--------------------------------')
        # print('-> max scoring for first pred:', sent_oba)
        # print('-> output sentence cos:', sentence_test_cos)

        # print(' ')
        # print('-> output second level:')
        # print(sentence2)

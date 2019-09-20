from process_movie_lines3 import embed_single
# from train_attention3 import infer_nmt
# from tensorflow.python.keras.models import load_model
# from tensorflow.python.keras.backend import set_session
# from layers.attention import AttentionLayer
from gensim.models import FastText
from bpemb import BPEmb
import numpy as np
import tensorflow as tf
import layers.attention as k_att
# from keras.utils import np_utils
import re

class NeuralBot():
    """
    for handling messages
    """
    def __init__(self, updater, model, graph):
        self.updater = updater
        self.graph = graph

        self.max_seq_len = 100
        self.n_vocab = 51
        self.diversity = 0.3
        self.model = model

        # self.session = tf.Session()

        self.char_to_int = {' ': 0, '!': 1, '"': 2, "'": 3, ',': 4, '-': 5, '.': 6, '/': 7, '0': 8, '1': 9, '2': 10, '3': 11, '4': 12, '5': 13, '6': 14, '7': 15, '8': 16, '9': 17, ':': 18, ';': 19, '<': 20, '>': 21, '?': 22, '@': 23, 'a': 24, 'b': 25, 'c': 26, 'd': 27, 'e': 28, 'f': 29, 'g': 30, 'h': 31, 'i': 32, 'j': 33, 'k': 34, 'l': 35, 'm': 36, 'n': 37, 'o': 38, 'p': 39, 'q': 40, 'r': 41, 's': 42, 't': 43, 'u': 44, 'v': 45, 'w': 46, 'x': 47, 'y': 48, 'z': 49, '~': 50}
        self.int_to_char = {0: ' ', 1: '!', 2: '"', 3: "'", 4: ',', 5: '-', 6: '.', 7: '/', 8: '0', 9: '1', 10: '2', 11: '3', 12: '4', 13: '5', 14: '6', 15: '7', 16: '8', 17: '9', 18: ':', 19: ';', 20: '<', 21: '>', 22: '?', 23: '@', 24: 'a', 25: 'b', 26: 'c', 27: 'd', 28: 'e', 29: 'f', 30: 'g', 31: 'h', 32: 'i', 33: 'j', 34: 'k', 35: 'l', 36: 'm', 37: 'n', 38: 'o', 39: 'p', 40: 'q', 41: 'r', 42: 's', 43: 't', 44: 'u', 45: 'v', 46: 'w', 47: 'x', 48: 'y', 49: 'z', 50: '~'}
        self.char_list = list(self.char_to_int.keys())

    def start(self):
        self.updater.start_polling()
        self.updater.idle()

    def hello(self, bot, update):
        print(update)
        update.message.reply_text(
            'Hello {}'.format(update.message.from_user.first_name))

    def respond(self, update, context):
        print('-> Message text:', update.message.text)
        print('-> User info:', update.message.from_user)
        print('-> User id:', update.message.from_user.id)
        name_lower = update.message.from_user.first_name.lower()
        print('-> From:', name_lower)
        print('-> Message context:', context)
        # if any(n_part in name_lower for n_part in self.huub_list):
        #     if not self.huub_seen:
        #         update.message.reply_text("HELLO KING HUBERT \n  \n   ヽ༼ ಠ益ಠ ༽ﾉ")
        #         self.huub_seen = True
        # elif any(n_part in name_lower for n_part in self.niels_list):
        #     print('-> Niels detect')
        #     if not self.niels_seen:
        #         update.message.reply_text("WELKOM MEESTER \n \n ╚╚|░☀▄☀░|╝╝ \n \n [-c°▥°]-c ")
        #         self.niels_seen = True

        output_sentence = self.process_chars(update.message.text)
        update.message.reply_text(output_sentence)

    def sample(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        # print('got preds')
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        # print('divide')
        # print(preds)
        probas = np.random.multinomial(1, preds, 1)
        # print('probas')
        return np.argmax(probas)

    def process_chars(self, line):
        print('-> processing chars...')
        # remove unwanted chars
        line = line.lower()
        line = ''.join([i for i in line if i in self.char_list])
        len_diff = len(line) - self.max_seq_len

        # pad with spaces
        if len_diff > 0:
            line = line[-self.max_seq_len:]
        elif len_diff < 0:
            padding = ' '*np.abs(len_diff)
            line = padding + line

        # pattern = [self.char_to_int[char] for char in line]

        res = '' + line
        generated = ''
        # print('pattern:', pattern)
        print( "Seed:")
        print(line)
        # print( "\"", ''.join([self.int_to_char[value] for value in pattern]), "\"")
        # generate characters
        # with self.graph.as_default():
        for i in range(400):
            try:
                x_pred = np.zeros((1, self.max_seq_len, self.n_vocab))
                for t, char in enumerate(res):
                    x_pred[0, t, self.char_to_int[char]] = 1.
            except Exception as ex:
                print('prediction error ', ex)
            try:
                preds = self.model.predict(x_pred, verbose=0)[0]
            except Exception as ex:
                print('error in model', ex)
                # preds = model.predict(x_pred, verbose=0)[0]
            try:
                next_index = self.sample(preds, self.diversity)
                next_char = self.int_to_char[next_index]
            except Exception as ex:
                print('sampling error', ex)
            res = res[1:] + next_char

            generated += next_char
            if next_char == '.':
                break
                # pattern.append(index)
                # pattern = pattern[1:len(pattern)]
        generated = re.sub('\:\:\:d', ':::D', generated)
        print('-> output sentence:', generated)
        return generated
        # except Exception as ex:
        #     print(ex)

    def clean_string(self, line):
        line = line.translate({ord(i): None for i in """[]'\""""})
        line = re.sub('<u>', '', line)
        line = re.sub('</u>', '', line)
        line = re.sub('<U>', '', line)
        line = re.sub('</U>', '', line)
        line = re.sub('\`', " ", line)
        line = re.sub('\-\-', ' ', line)
        line = re.sub('\. \. \.', ' . ', line)
        line = re.sub('\.  \.  \.', ' . ', line)
        line = re.sub(' +', ' ', line)
        line = line.lower()
        return line

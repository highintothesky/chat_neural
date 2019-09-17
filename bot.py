from process_movie_lines3 import embed_single
# from train_attention3 import infer_nmt
# from tensorflow.python.keras.models import load_model
# from tensorflow.python.keras.backend import set_session
from layers.attention import AttentionLayer
from gensim.models import FastText
from bpemb import BPEmb
import numpy as np
import tensorflow as tf
import layers.attention as k_att
from keras.utils import np_utils
import re

class NeuralBot():
    """
    for handling messages
    """
    def __init__(self, updater, model, graph):
        self.updater = updater
        self.graph = graph
        self.huub_seen = False
        self.niels_seen = False
        self.huub_list = ['huub', 'hubert']
        self.niels_list = ['niels']
        self.max_sen_len = 20
        self.model = model

        self.bpemb_en = BPEmb(lang="en", dim=100, vs=100000)
        self.session = tf.Session()

        self.char_to_int = {' ': 0, '!': 1, '%': 2, '&': 3, "'": 4, ',': 5, '-': 6, '.': 7, '/': 8, '0': 9, '1': 10, '2': 11, '3': 12, '4': 13, '5': 14, '6': 15, '7': 16, '8': 17, '9': 18, ':': 19, ';': 20, '<': 21, '>': 22, '?': 23, 'a': 24, 'b': 25, 'c': 26, 'd': 27, 'e': 28, 'f': 29, 'g': 30, 'h': 31, 'i': 32, 'j': 33, 'k': 34, 'l': 35, 'm': 36, 'n': 37, 'o': 38, 'p': 39, 'q': 40, 'r': 41, 's': 42, 't': 43, 'u': 44, 'v': 45, 'w': 46, 'x': 47, 'y': 48, 'z': 49, '~': 50, '—': 51}
        self.int_to_char = {0: ' ', 1: '!', 2: '%', 3: '&', 4: "'", 5: ',', 6: '-', 7: '.', 8: '/', 9: '0', 10: '1', 11: '2', 12: '3', 13: '4', 14: '5', 15: '6', 16: '7', 17: '8', 18: '9', 19: ':', 20: ';', 21: '<', 22: '>', 23: '?', 24: 'a', 25: 'b', 26: 'c', 27: 'd', 28: 'e', 29: 'f', 30: 'g', 31: 'h', 32: 'i', 33: 'j', 34: 'k', 35: 'l', 36: 'm', 37: 'n', 38: 'o', 39: 'p', 40: 'q', 41: 'r', 42: 's', 43: 't', 44: 'u', 45: 'v', 46: 'w', 47: 'x', 48: 'y', 49: 'z', 50: '~', 51: '—'}
        self.char_list = list(self.char_to_int.keys())
        self.n_vocab = 52

    def start(self):
        self.updater.start_polling()
        self.updater.idle()

    def hello(self, bot, update):
        # print(update)
        update.message.reply_text(
            'Hello {}'.format(update.message.from_user.first_name))

    def respond(self, update, context):
        print('-> Message text:', update.message.text)
        name_lower = update.message.from_user.first_name.lower()
        print('-> From:', name_lower)
        print('-> Message context:', context)
        if any(n_part in name_lower for n_part in self.huub_list):
            if not self.huub_seen:
                update.message.reply_text("HELLO KING HUBERT \n  \n   ヽ༼ ಠ益ಠ ༽ﾉ")
                self.huub_seen = True
        elif any(n_part in name_lower for n_part in self.niels_list):
            print('-> Niels detect')
            if not self.niels_seen:
                update.message.reply_text("WELKOM MEESTER \n \n ╚╚|░☀▄☀░|╝╝ \n \n [-c°▥°]-c ")
                self.niels_seen = True

        output_sentence = self.process_chars(update.message.text)
        update.message.reply_text(output_sentence)

    def get_neural_response(self, input_string):
        cleaned = self.clean_string(input_string)
        # restrict its length
        cleaned_list = cleaned.split()
        print(cleaned_list)
        # cleaned = ' '.join([word_i for word_i in cleaned_list[-30:]])
        sentences = [cleaned]
        print('-> getting neural response')
        try:
            # global graph
            # graph = tf.get_default_graph()
            mat = embed_single(cleaned, self.bpemb_en, seq_length=30)
            print('-> mat shape:')
            print(mat.shape)
            test_in_seq = np.expand_dims(mat[:,:], 0)
            print('-> Test sentence:')
            print(sentences[0])
            out_sentence = ''
            print(test_in_seq.shape)
        except Exception as ex:
            print(ex)
        try:
            with self.graph.as_default():
                sentence_test = ''
                # this_len = 0
                test_in_seq = np.expand_dims(mat, 0)
                seq_total = test_in_seq
                sent_test = ''
                print('-> Predicting...')
                for j in range(30):

                    arr = self.model.predict(test_in_seq)
                    word = bpemb_en.most_similar(positive=[arr[0,:]])[0][0] + ' '
                    if word[0] == '▁':
                        word = word[1:]
                    sentence_test += word
                    arr = np.expand_dims(arr, 0)

                    test_in_seq = test_in_seq[:,1:,:]
                    test_in_seq = np.concatenate((test_in_seq, arr), axis=1)

                    seq_total = np.concatenate((seq_total, arr), axis=1)

                out_arr = seq_total[0,30:,:]
                sent_oba_sc = self.bpemb_en.most_similar(positive=out_arr)

                sent_oba = ''
                for word_sc in sent_oba_sc:
                    word = word_sc[0]
                    if word[0] == '▁':
                        word = word[1:]
                    sent_oba += word + ' '
                print('-> Done predicting')
                # sentence_sc = self.bpemb_en.most_similar(test_in_seq[0,:,:])
                # first character is always _
                # out_sentence = ' '.join(word[0][1:] for word in sentence_sc)
                print('-> Output sentence:')
                print(sentence_test)
                # print('-> inference Test:', sent_test)
            # out_sentence = re.sub(r"(.+?)\2+", r"\2", out_sentence)
        except Exception as ex:
            print(ex)
        return sentence_test

    def process_chars(self, line):
        print('-> processing chars...')
        # remove unwanted chars
        try:
            line = line.lower()
            line = ''.join([i for i in line if i in self.char_list])
            len_diff = len(line) - 100

            # pad with spaces
            if len_diff > 0:
                line = line[-100:]
            elif len_diff < 0:
                padding = ' '*np.abs(len_diff)
                line = padding + line

            pattern = [self.char_to_int[char] for char in line]
            # normalize
            # X_arr = np.array(X_list)/float(self.n_vocab)
            # X_in = np.reshape(X_arr, (1, 100, 1))
            res = ''
            print('pattern:', pattern)
            print( "Seed:")
            print( "\"", ''.join([self.int_to_char[value] for value in pattern]), "\"")
            # generate characters
            with self.graph.as_default():
                for i in range(100):
                    x = np.reshape(pattern, (1, len(pattern), 1))
                    x = x / float(self.n_vocab)
                    prediction = self.model.predict(x, verbose=0)
                    index = np.argmax(prediction)
                    # print('index pred:', index)
                    result = self.int_to_char[index]
                    # seq_in = [self.int_to_char[value] for value in pattern]
                    # sys.stdout.write(result)
                    res += result
                    pattern.append(index)
                    pattern = pattern[1:len(pattern)]

            print('-> output sentence:', res)
            return res
        except Exception as ex:
            print(ex)

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

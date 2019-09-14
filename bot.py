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

        output_sentence = self.get_neural_response(update.message.text)
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

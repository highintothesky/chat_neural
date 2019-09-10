from process_movie_lines2 import sentences_to_matrix
# from train_attention3 import infer_nmt
# from tensorflow.python.keras.models import load_model
# from tensorflow.python.keras.backend import set_session
from layers.attention import AttentionLayer
from gensim.models import FastText
import numpy as np
import tensorflow as tf
import layers.attention as k_att

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
        # global sess
        # sess = tf.Session()
        # set_session(sess)
        # global graph
        # graph = tf.get_default_graph()
        # with graph.as_default():
        #     self.att_model = keras.models.load_model('models/att1_full.h5',
        #                                             custom_objects=SeqSelfAttention.get_custom_objects())
        self.ft_model = FastText.load('models/fasttext2')
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
                update.message.reply_text("HEY HUUB-SAN \n ヽ༼ ಠ益ಠ ༽ﾉ")
                self.huub_seen = True
        elif any(n_part in name_lower for n_part in self.niels_list):
            print('-> Niels detect')
            if not self.niels_seen:
                update.message.reply_text("WELKOM MEESTER \n ┗(｀Дﾟ┗(｀ﾟДﾟ´)┛ﾟД´)┛")
                self.niels_seen = True
        # else:


        output_sentence = self.get_neural_response(update.message.text)
        update.message.reply_text(output_sentence)
        # print('-> Replying...')
        # update.message.reply_text(update.message.text)

    def get_neural_response(self, input_string):
        sentences = [input_string]
        print('-> getting neural response')
        try:
            # global graph
            # graph = tf.get_default_graph()
            mat = sentences_to_matrix(sentences, self.ft_model)
            print('-> mat shape:')
            print(mat.shape)
            test_in_seq = np.expand_dims(mat[0,:,:], 0)
            print('-> Test sentence:')
            print(sentences[0])
            out_sentence = ''
            print(test_in_seq.shape)
        except Exception as ex:
            print(ex)
        try:
            for j in range(15):
                with self.graph.as_default():
                    # with self.session.as_default():
                    arr = self.model.predict(test_in_seq)

                arr = np.expand_dims(arr, 0)
                # print(arr.shape)
                word = self.ft_model.most_similar(positive=[arr[0,0,:]])[0][0]
                # print(word)
                out_sentence += word + ' '

                test_in_seq = test_in_seq[:,1:,:]
                test_in_seq = np.concatenate((test_in_seq, arr), axis=1)
        except Exception as ex:
            print(ex)
        return out_sentence

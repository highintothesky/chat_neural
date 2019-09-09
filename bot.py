from process_movie_lines import sentences_to_matrix
from train_attention import infer_nmt
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.backend import set_session
from layers.attention import AttentionLayer
from gensim.models import FastText
import numpy as np
import tensorflow as tf
import layers.attention as k_att

class NeuralBot():
    """
    for handling messages
    """
    def __init__(self, updater):
        self.updater = updater
        self.huub_seen = False
        self.niels_seen = False
        self.huub_list = ['huub', 'hubert']
        self.niels_list = ['niels']
        self.max_sen_len = 30
        global sess
        sess = tf.Session()
        set_session(sess)
        global graph
        graph = tf.get_default_graph()
        with graph.as_default():
            self.att_model = load_model('models/att1_full.h5',
                                   custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform,
                                                   "AttentionLayer": k_att.AttentionLayer})
            # self.infer_enc_model = load_model('models/att1_enc.h5',
            #                        custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform,
            #                                        "AttentionLayer": k_att.AttentionLayer})
            # self.infer_dec_model = load_model('models/att1_dec.h5',
            #                        custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform,
            #                                        "AttentionLayer": k_att.AttentionLayer})
            # self.infer_enc_model.compile(optimizer='adam', loss='mse')
            # self.infer_dec_model.compile(optimizer='adam', loss='mse')
        # print(self.infer_enc_model.summary())
        self.ft_model = FastText.load('models/fasttext2')

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
        sentences = [update.message.text]
        mat = sentences_to_matrix(sentences, self.ft_model, 30)
        print('-> mat shape:')
        print(mat.shape)
        test_en_seq = np.expand_dims(mat[0,:,:], 0)
        print('-> Test sentence:')
        print(sentences[0])
        print(test_en_seq.shape)
        # global sess
        # global graph
        # graph = tf.get_default_graph()
        with graph.as_default():
            try:
                in_arr = np.zeros((512,30,100))
                out_arr = np.zeros((512,29,100))
                print('-> made out arr')
                sent_mat = sentences_to_matrix(sentences, self.ft_model, self.max_sen_len)
                start_mat = sentences_to_matrix(['sos'], self.ft_model, self.max_sen_len-1)
                in_arr[0,:,:] = sent_mat
                out_arr[0,:,:] = start_mat
                print('-> got word embeddings...')
                pred_arr = self.att_model.predict([in_arr, out_arr])
                out_str = ''
                for i in range(29):
                    this_word = ft_model.most_similar(positive=[a_batch[0,i,:]])[0][0]
                    print(this_word)
                    out_str += this_word + ' '
                # test_fr, attn_weights = infer_nmt(
                #     encoder_model=self.infer_enc_model, decoder_model=self.infer_dec_model,
                #     test_en_seq=test_en_seq, emb_len = 100, ft_model = self.ft_model)
                print('-> Done. output text:')
                print(out_str)
                update.message.reply_text(out_str)
            except Exception as ex:
                print(ex)
                update.message.reply_text(update.message.text)
        # print('-> Replying...')
        # update.message.reply_text(update.message.text)

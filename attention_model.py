import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import get_file, to_categorical, CustomObjectScope
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input, Bidirectional, Flatten, Dense, GRU, Dropout, BatchNormalization
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
# from https://github.com/ongunuzaymacar/attention-mechanisms
from attention.layers import Attention, SelfAttention

class AttentionModel():
    """
    Attention/GRU based text generation model
    """

    def __init__(self,
                 checkpoint_path,
                 seq_len = 100,
                 emb_len = 51,
                 rnn_size = 256,
                 rnn_style = 'GRU',
                 dropout_rate = 0.1,
                 bidirectional = False,
                 load_model = False):
        self.seq_len = seq_len
        self.emb_len = emb_len
        self.rnn_size = rnn_size
        self.rnn_style = rnn_style
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.checkpoint_path = checkpoint_path
        self.json_path = checkpoint_path + '/model.json'

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        # build the model
        if not load_model:
            # build from the args
            self.build_model()
        elif load_model:
            # load arch from json, build
            self.from_json()

    def build_model(self):
        """
        Build the model based on the params
        """
        print('-> Building model')
        inputs = Input(shape=(self.seq_len, self.emb_len), name = 'inputs')

        # add a RNN layer
        if self.rnn_style == 'GRU' and self.bidirectional:
            encoder_output, hidden_state = Bidirectional(GRU(units=self.rnn_size,
                                                             return_sequences=True,
                                                             return_state=True,
                                                             name = 'GRU bi'))(inputs)
        elif self.rnn_style == 'GRU' and not self.bidirectional:
            encoder_output, hidden_state = GRU(units=self.rnn_size,
                                               return_sequences=True,
                                               return_state=True,
                                               name = 'GRU')(inputs)
        elif self.rnn_style == 'CuDNNLSTM':
            encoder_output, hidden_state, c_state = CuDNNLSTM(units=self.rnn_size,
                                                              return_sequences=True,
                                                              return_state=True,
                                                              name = 'CuDNNLSTM')(inputs)
        # concat for use as input
        attention_input = [encoder_output, hidden_state]

        # get attention and flatten
        encoder_output, attention_weights = Attention(context='many-to-one',
                                                      alignment_type='local-p*',
                                                      window_width=25,
                                                      name = 'Attention')(attention_input)
        encoder_output = Flatten(name = 'Flatten')(encoder_output)

        # against overfitting etc
        x = BatchNormalization(name = 'BatchNormalization')(encoder_output)
        x = Dropout(self.dropout_rate, name = 'Dropout')(x)
        predictions = Dense(self.emb_len,
                            activation='softmax',
                            name = 'Dense')(x)

        # define the model
        self.model = Model(inputs=inputs, outputs=predictions)

        print(self.model.summary())

    def save_config(self):
        """
        Save the model hyperparams such as style and layer size
        """
        dump_dict = {'seq_len': self.seq_len,
                     'emb_len': self.emb_len,
                     'rnn_size': self.rnn_size,
                     'rnn_style': self.rnn_style,
                     'dropout_rate': self.dropout_rate,
                     'bidirectional': self.bidirectional}
        with open(self.json_path, 'w') as fp:
            json.dump(dump_dict, fp)

    def from_json(self):
        """
        load from json params
        """
        with open(self.json_path, 'r') as fp:
            arch_dict = json.load(fp)
        self.seq_len = arch_dict['seq_len']
        self.emb_len = arch_dict['emb_len']
        self.rnn_size = arch_dict['rnn_size']
        self.rnn_style = arch_dict['rnn_style']
        self.dropout_rate = arch_dict['dropout_rate']
        self.bidirectional = arch_dict['bidirectional']

        self.build_model()

        print('-> Loading model weights...')
        self.model.load_weights(self.checkpoint_path)
        print('-> Done.')

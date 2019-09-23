
from telegram.ext import CallbackContext
from telegram import Update

import numpy as np
import tensorflow as tf
import re

class NeuralBot():
    """
    for handling messages
    """
    def __init__(self, updater, model, graph):
        self.updater = updater
        self.graph = graph

        # max length of prediction, set with /set_length
        self.max_seq_len = 300
        self.input_len = 100
        self.n_vocab = 51

        # default diversity. can be changed with /set_diversity
        self.diversity = 0.4
        # model stops after a few periods
        self.periods = 5
        self.stop_signs = """!?.:;"""
        self.model = model

        self.char_to_int = {' ': 0, '!': 1, '"': 2, "'": 3, ',': 4, '-': 5, '.': 6, '/': 7, '0': 8, '1': 9, '2': 10, '3': 11, '4': 12, '5': 13, '6': 14, '7': 15, '8': 16, '9': 17, ':': 18, ';': 19, '<': 20, '>': 21, '?': 22, '@': 23, 'a': 24, 'b': 25, 'c': 26, 'd': 27, 'e': 28, 'f': 29, 'g': 30, 'h': 31, 'i': 32, 'j': 33, 'k': 34, 'l': 35, 'm': 36, 'n': 37, 'o': 38, 'p': 39, 'q': 40, 'r': 41, 's': 42, 't': 43, 'u': 44, 'v': 45, 'w': 46, 'x': 47, 'y': 48, 'z': 49, '~': 50}
        self.int_to_char = {0: ' ', 1: '!', 2: '"', 3: "'", 4: ',', 5: '-', 6: '.', 7: '/', 8: '0', 9: '1', 10: '2', 11: '3', 12: '4', 13: '5', 14: '6', 15: '7', 16: '8', 17: '9', 18: ':', 19: ';', 20: '<', 21: '>', 22: '?', 23: '@', 24: 'a', 25: 'b', 26: 'c', 27: 'd', 28: 'e', 29: 'f', 30: 'g', 31: 'h', 32: 'i', 33: 'j', 34: 'k', 35: 'l', 36: 'm', 37: 'n', 38: 'o', 39: 'p', 40: 'q', 41: 'r', 42: 's', 43: 't', 44: 'u', 45: 'v', 46: 'w', 47: 'x', 48: 'y', 49: 'z', 50: '~'}
        self.char_list = list(self.char_to_int.keys())

        # log all communication to the bot
        self.log_file = open('data/bot_conversations.txt', 'a')

    def start(self):
        self.updater.start_polling()
        self.updater.idle()

    def hello(self, update, context):
        update.message.reply_text(
            'Hello {}'.format(update.message.from_user.first_name))

    def get_pred_len(self, update, context):
        # get the prediction char length max
        try:
            max_seq_len = context.user_data['max_seq_len']
        except KeyError:
            context.user_data['max_seq_len'] = self.max_seq_len
            max_seq_len = self.max_seq_len
        return max_seq_len

    def get_div(self, update, context):
        try:
            div = context.user_data['diversity']
        except KeyError:
            context.user_data['diversity'] = self.diversity
            div = self.diversity
        return div

    def get_old_input(self, context):
        # get the user's past sentences
        try:
            previous = context.user_data['old_input']
        except KeyError:
            context.user_data['old_input'] = ' '
            previous = ' '
        return previous

    def set_diversity(self, update: Update, context: CallbackContext):
        print('-> Setting diversity, update:', update.message.text)
        try:
            val_str = update.message.text.split()[1]
            context.user_data['diversity'] = float(val_str)
            update.message.reply_text(
                'Diversity set to {}'.format(context.user_data['diversity'])
            )
        except Exception as ex:
            print('-> Failed setting diversity:', ex)
            update.message.reply_text(
                'Failed setting diversity. Please format like this: /set_diversity 0.3'
            )

    def set_pred_len(self, update, context):
        print('-> Setting max_seq_len, update:', update.message.text)
        try:
            val_str = update.message.text.split()[1]
            context.user_data['max_seq_len'] = int(val_str)
            update.message.reply_text(
                'Maximum sequence length set to {} chars'.format(context.user_data['max_seq_len'])
            )
        except Exception as ex:
            print('-> Failed setting max_seq_len:', ex)
            update.message.reply_text(
                'Failed setting sequence length. Please format like this: /set_length 300'
            )

    def set_old_input(self, context, old_input):
        # remember what was said by the user before
        if len(old_input) > self.input_len:
            # truncate
            old_input = old_input[-self.input_len:]
        context.user_data['old_input'] = old_input

    def respond(self, update, context):
        print('-> Message text:', update.message.text)
        print('-> User info:', update.message.from_user)
        print('-> User id:', update.message.from_user.id)
        name_lower = update.message.from_user.first_name.lower()
        print('-> From:', name_lower)
        print('-> Message context:', context)

        # log the message
        self.log_file.write(update.message.text + '\n')
        self.log_file.flush()

        diversity = self.get_div(update, context)
        out_len = self.get_pred_len(update, context)

        old_input = self.get_old_input(context)

        # add a period so the sentences don't just concatenate
        if not old_input[-1] in self.stop_signs:
            old_input += '.'

        in_sent = old_input + ' ' + update.message.text
        self.set_old_input(context, in_sent)

        output_sentence = self.process_chars(in_sent, diversity, out_len)
        update.message.reply_text(output_sentence)

    def sample(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature

        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)

        return np.argmax(probas)

    def process_chars(self, line, diversity, out_len):
        print('-> processing chars...')
        # remove unwanted chars
        line = line.lower()
        line = ''.join([i for i in line if i in self.char_list])
        len_diff = len(line) - self.input_len

        # pad with spaces
        if len_diff > 0:
            line = line[-self.input_len:]
        elif len_diff < 0:
            # repeat it
            line = (line + ' ')*50
            line = line[-self.input_len:]

        res = '' + line
        generated = ''

        print( "Seed:")
        print(line)

        # count the number of periods we've seen
        periods = 0

        # generate characters
        for i in range(out_len):
            try:
                x_pred = np.zeros((1, self.input_len, self.n_vocab))
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
                next_index = self.sample(preds, diversity)
                next_char = self.int_to_char[next_index]
            except Exception as ex:
                print('sampling error', ex)
            res = res[1:] + next_char

            generated += next_char
            if next_char == '.':
                periods += 1
                if periods > self.periods:
                    break

        generated = re.sub('\:\:\:d', ':::D', generated)
        print('-> output sentence:', generated)
        return generated

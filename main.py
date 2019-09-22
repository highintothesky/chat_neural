import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
# import getopt
import tensorflow as tf
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, PicklePersistence
from bot import NeuralBot
from attention_model import AttentionModel

if __name__ == '__main__':
    debug = 0
    if len(sys.argv) > 1:
        if sys.argv[1] == '--debug':
            debug = 1

    print('-> Starting Bot!')
    f = open("token.txt", "r")
    token = f.read().strip()


    if debug == 0:
        checkpoint_path = "models/char_att6/"
        global model
        model = AttentionModel(checkpoint_path = checkpoint_path,
                               load_model = True).model
    elif debug == 1:
        model = 'yolo'
    # with CustomObjectScope({'SeqSelfAttention': SeqSelfAttention,
    #                         'MultiHead': MultiHead,
    #                         'root_mean_squared_error': root_mean_squared_error}):
    #     model = keras.models.load_model('models/char_18_epoch_5.h5')
    # global graph
    # graph = tf.get_default_graph()
    graph = None

    pp = PicklePersistence(filename='data/conversationbot')
    updater = Updater(token, persistence=pp, use_context=True)
    bot = NeuralBot(updater, model, graph)

    updater.dispatcher.add_handler(CommandHandler('hello', bot.hello))
    updater.dispatcher.add_handler(CommandHandler('set_diversity', bot.set_diversity))
    updater.dispatcher.add_handler(CommandHandler('set_length', bot.set_pred_len))
    updater.dispatcher.add_handler(MessageHandler(Filters.text, bot.respond))

    bot.start()

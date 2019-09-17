import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from bot import NeuralBot
import keras
from keras_self_attention import SeqSelfAttention
from train_functional import root_mean_squared_error
from keras_multi_head import MultiHead
from keras.utils import CustomObjectScope
import tensorflow as tf

if __name__ == '__main__':
    print('-> Starting Bot!')
    f = open("token.txt", "r")
    token = f.read().strip()

    with CustomObjectScope({'SeqSelfAttention': SeqSelfAttention,
                            'MultiHead': MultiHead,
                            'root_mean_squared_error': root_mean_squared_error}):
        model = keras.models.load_model('models/char_10_epoch_4.h5')
    global graph
    graph = tf.get_default_graph()

    updater = Updater(token, use_context=True)
    bot = NeuralBot(updater, model, graph)

    updater.dispatcher.add_handler(CommandHandler('hello', bot.hello))
    updater.dispatcher.add_handler(MessageHandler(Filters.text, bot.respond))

    bot.start()

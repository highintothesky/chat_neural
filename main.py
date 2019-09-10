from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from bot import NeuralBot
import keras
from keras_self_attention import SeqSelfAttention
import tensorflow as tf

if __name__ == '__main__':
    print('-> Starting Bot!')
    f = open("token.txt", "r")
    token = f.read().strip()

    model = keras.models.load_model('models/att5_full.h5',
                                    custom_objects=SeqSelfAttention.get_custom_objects())
    global graph
    graph = tf.get_default_graph()

    updater = Updater(token, use_context=True)
    bot = NeuralBot(updater, model, graph)

    updater.dispatcher.add_handler(CommandHandler('hello', bot.hello))
    updater.dispatcher.add_handler(MessageHandler(Filters.text, bot.respond))

    bot.start()

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from bot import NeuralBot


if __name__ == '__main__':
    print('-> Starting Bot!')
    f = open("token.txt", "r")
    token = f.read().strip()

    # load the networks in main
    print('-> Loading att model')
    att_model = load_model('models/att1_full.h5',
                           custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform,
                                           "AttentionLayer": k_att.AttentionLayer})
    print('-> Loading enc model')
    infer_enc_model = load_model('models/att1_enc.h5',
                           custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform,
                                           "AttentionLayer": k_att.AttentionLayer})
    print('-> Loading dec model')

    infer_dec_model = load_model('models/att1_dec.h5',
                           custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform,
                                           "AttentionLayer": k_att.AttentionLayer})
    ft_model = FastText.load('models/fasttext2')

    updater = Updater(token, use_context=True)
    bot = NeuralBot(updater)

    updater.dispatcher.add_handler(CommandHandler('hello', bot.hello))
    updater.dispatcher.add_handler(MessageHandler(Filters.text, bot.respond))

    bot.start()

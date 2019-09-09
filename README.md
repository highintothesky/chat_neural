# chat_neural
Chatbot running an attention-based neural network trained on movie dialog

This is a telegram bot, running a tf.keras attention model. The attention layers were shamelessly lifted from https://github.com/thushv89/attention_keras . I use Fasttext word embeddings, and https://github.com/python-telegram-bot/python-telegram-bot

Keep in mind that this is just a hobby project, so the code is very messy and I will probably never clean it up.

you will need your token.txt, movie dialog data in ./data, and a models folder for storing all those hdf5 objects etc.

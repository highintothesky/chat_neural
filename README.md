# chat_neural
Chatbot running an attention-based neural network trained on movie dialog

This is a telegram bot, running a tf.keras attention model. The attention layers were shamelessly lifted from https://github.com/ongunuzaymacar/attention-mechanisms . I use binary character embedding, and https://github.com/python-telegram-bot/python-telegram-bot

put your text data in `data/mixed.txt`, then:

```python3 process_to_h5.py```

to generate some hdf5 files from your text data make sure the references to these files are given to the batchgenerator before you run 

```python3 train_chars_generator.py```

to train a model. make sure you have a `models` folder for storing the model info and weights. you can test your model using `infer_char.py` and run the telegram bot using `main.py` (you will need your token.txt containing your telegram api token)


Keep in mind that this is just a hobby project, so the code is very messy and I will probably never clean it up.


# train using https://pypi.org/project/keras-self-attention/
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import re
# import tensorflow as tf
# from tensorflow import keras
# import h5py
# import numpy as np

import os
import re
import argparse
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy

from attention_model import AttentionModel
from batch_generator import BatchGenerator

def perplexity(labels, logits):
    """Calculates perplexity metric = 2^(entropy) or e^(entropy)"""
    return pow(2, categorical_crossentropy(y_true=labels, y_pred=logits))

def main():
    h5_list = [h5py.File('data/processed0.h5', 'r'), h5py.File('data/processed1.h5', 'r'), h5py.File('data/processed3.h5', 'r'), h5py.File('data/processed4.h5', 'r')]
    h5_list_test = [h5py.File('data/processed2.h5', 'r')]

    batch_size = 100

    bg_train = BatchGenerator(h5_list, batch_size)
    bg_test = BatchGenerator(h5_list_test, batch_size) #, maxlen = 400000)

    # if we want to change batch size during training
    # dynamic_batch = True
    dynamic_batch = False

    n_epochs = 30
    opt = keras.optimizers.Adam()
    # opt = keras.optimizers.Adadelta()
    # opt = keras.optimizers.RMSprop(lr=0.001)

    # which iteration of models to load
    # next_it = 2
    #     # encoder_output, attention_weights = SelfAttention(size=50,
    #     #                                                   num_hops=16,
    #     #                                                   use_penalization=False)(x)

    checkpoint_path = "models/char_att7/"

    # Create checkpoint callback
    cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                  monitor='val_accuracy',
                                                  verbose=1,
                                                  save_weights_only=True,
                                                  mode='max',
                                                  save_best_only=True)

    load_model = False
    # load_model = True

    am = AttentionModel(checkpoint_path = checkpoint_path,
                        rnn_size = 512,
                        rnn_style = 'GRU',#'CuDNNLSTM',
                        # bidirectional = True,
                        dropout_rate = 0.4,
                        load_model = load_model)
    # am.model.save(checkpoint_path + 'model.h5')
    # am.build_model()
    am.model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy', perplexity, categorical_accuracy],
    )
    # print(am.model.summary())
    am.save_config()

    # am.model.fit(X_train,
    #              y_train,
    #              batch_size = batch_size,
    #              validation_data=(X_test, y_test),
    #              callbacks = [cp_callback],
    #              epochs=n_epochs)
     # fit using the batch generators
    am.model.fit_generator(bg_train,
                           validation_data=bg_test,
                           callbacks = [cp_callback],
                           # use_multiprocessing=True,
                           # workers=4,
                           epochs=n_epochs)
    # for i in range(n_epochs):
    #     full_new_path = 'models/char_att_{}_epoch_{}.h5'.format(next_it, i)
    #     # set batch size dynamically
    #     if dynamic_batch:
    #         if i == 1:
    #             batch_size = 128
    #         elif i == 2:
    #             batch_size = 64
    #         else:
    #             batch_size = 32

                            # validation_data=(oba_in_tests, oba_out_test))
        # print('-> Saving to', full_new_path)
        # with CustomObjectScope({'SelfAttention': SelfAttention,
        #                         'Attention': Attention,
        #                         'perplexity': perplexity}):
        # model.save(full_new_path)

if __name__ == '__main__':
    main()

# test the trained model v2
import numpy as np
import keras
import keras_self_attention
from process_movie_lines import sentences_to_matrix
from tensorflow.python.keras.models import load_model

from keras_self_attention import SeqSelfAttention
from gensim.models import FastText


if __name__ == '__main__':
    print('-> Loading att model')
    att_model = keras.models.load_model('models/att4_full.h5',
                                        custom_objects=SeqSelfAttention.get_custom_objects())
    ft_model = FastText.load('models/fasttext2')

    print(att_model.summary())
    sentences = ["sos hey, what's up? eos",
                 "sos kiss my ass eos",
                 "sos Are you really a robot? I can't believe it! eos"]
    mat = sentences_to_matrix(sentences, ft_model, 30)
    print('-> test mat shape:', mat.shape)

    for i in range(len(sentences)):
        test_in_seq = np.expand_dims(mat[i,:,:], 0)

        print('-> Test sentence:')
        print(sentences[i])
        out_arr = att_model.predict(test_in_seq)

        print(out_arr.shape)
        print(out_arr)
        out_sentence = ''
        for j in range(29):
            word = ft_model.most_similar(positive=[out_arr[0,j,:]])[0][0]
            out_sentence += word + ' '

        print('-> Done. output text:')
        print(out_sentence)

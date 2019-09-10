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
    att_model = keras.models.load_model('models/att3_full.h5',
                                        custom_objects=SeqSelfAttention.get_custom_objects())
    ft_model = FastText.load('models/fasttext2')

    print(att_model.summary())
    sentences = ["sos hey, what's up? eos",
                 "sos kiss my ass eos",
                 "sos Are you really a robot? I can't believe it! eos",
                 "sos Gosh if only we could find Kat a boyfriend  eos"]
    mat = sentences_to_matrix(sentences, ft_model, 20)
    print('-> test mat shape:', mat.shape)
    # normalize
    # -> Lowest value in data:
    # -16.463590621948242
    # -> highest value in data (after adding):
    # 33.590084075927734
    # dat_min = -16.463590621948242
    # dat_max = 33.590084075927734
    # mat -= dat_min
    # mat = mat/dat_max



    for i in range(len(sentences)):
        test_in_seq = np.expand_dims(mat[i,:,:], 0)

        print('-> Test sentence:')
        print(sentences[i])
        print(test_in_seq.shape)
        print(test_in_seq)
        out_arr = att_model.predict(test_in_seq)
        print(out_arr.shape)
        print(out_arr)
        # now denormalize
        out_arr = (out_arr * dat_max)+dat_min

        # print(out_arr)
        out_sentence = ''
        for j in range(1,20):
            word = ft_model.most_similar(positive=[out_arr[0,j,:]])[0][0]
            out_sentence += word + ' '

        print('-> Done. output text:')
        print(out_sentence)

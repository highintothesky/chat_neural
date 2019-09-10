# test the trained model v3
import re
import numpy as np
import keras
# import keras_self_attention
from keras_multi_head import MultiHead
from keras.utils import CustomObjectScope
from process_movie_lines2 import sentence_to_matrix, sentences_to_matrix
from keras.models import load_model

from keras_self_attention import SeqSelfAttention
from gensim.models import FastText


if __name__ == '__main__':
    print('-> Loading att model')

    with CustomObjectScope({'SeqSelfAttention': SeqSelfAttention,
                            'MultiHead': MultiHead}):
        model = keras.models.load_model('models/att8_full.h5')
    # model = keras.models.load_model('models/att6_full.h5',
    #                                 custom_objects={"SeqSelfAttention": SeqSelfAttention.get_custom_objects(),
    #                                 "MultiHead": MultiHead})
    ft_model = FastText.load('models/fasttext2')

    print(model.summary())
    sentences = ["hey, what's up?",
                 "kiss my ass",
                 "Are you really a robot? I can't believe it! now take this long string and do something with it",
                 "Gosh if only we could find Kat a boyfriend"]
    for idx, line in enumerate(sentences):
        line = re.sub('<u>', '', line)
        line = re.sub('</u>', '', line)
        line = re.sub('<U>', '', line)
        line = re.sub('</U>', '', line)
        line = re.sub('\`', " ", line)
        line = re.sub('\-\-', ' ', line)
        line = re.sub('\.', ' . ', line)
        line = re.sub('\,', ' , ', line)
        # line = re.sub('\.\.\.', ' ', line)
        line = re.sub('\. \. \.', ' . ', line)
        line = re.sub('\.  \.  \.', ' . ', line)
        # line = re.sub('\?\.', '?', line)
        line = re.sub('\?', ' ? ', line)
        line = re.sub('\!', ' ! ', line)
        line = re.sub('\;', ' , ', line)
        line = re.sub('\:', ' . ', line)
        line = re.sub(' +', ' ', line)
        line = line.lower()
        sentences[idx] = line
    mat = sentences_to_matrix(sentences, ft_model)
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
        sentence = ''
        word = ''
        this_len = 0
        test_in_seq = np.expand_dims(mat[i,:,:], 0)
        # print(test_in_seq.shape)
        while word != '.' and this_len < 30:
            this_len += 1
            arr = model.predict(test_in_seq)

            arr = np.expand_dims(arr, 0)
            # print(arr.shape)
            word = ft_model.most_similar(positive=[arr[0,0,:]])[0][0]
            # print(word)
            # print('-'+word+'-')
            sentence += word + ' '

            test_in_seq = test_in_seq[:,1:,:]
            test_in_seq = np.concatenate((test_in_seq, arr), axis=1)
            # print(arr)
        print(sentences[i])
        # sentence = re.sub(r"(.+?)\1+", r"\1", sentence)

        print(sentence)

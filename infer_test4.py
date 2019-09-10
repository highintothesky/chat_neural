# test the trained model v3
import re
import numpy as np
import keras
# import keras_self_attention
from keras_multi_head import MultiHead
from keras.utils import CustomObjectScope
from process_movie_lines3 import embed_single
from keras.models import load_model
from bpemb import BPEmb

from keras_self_attention import SeqSelfAttention
from gensim.models import FastText


if __name__ == '__main__':
    print('-> Loading att model')

    with CustomObjectScope({'SeqSelfAttention': SeqSelfAttention,
                            'MultiHead': MultiHead}):
        model = keras.models.load_model('models/att11_full.h5')

    bpemb_en = BPEmb(lang="en", dim=50)
    seq_length = 30

    print(model.summary())
    sentences = ["hey, what's up?",
                 "kiss my ass",
                 "Are you really a robot? I can't believe it! now take this long string and do something with it",
                 "Gosh if only we could find Kat a boyfriend"]
    embeddings = []
    for idx, line in enumerate(sentences):
        line = line.translate({ord(i): None for i in """[]'\""""})
        line = re.sub('<u>', '', line)
        line = re.sub('</u>', '', line)
        line = re.sub('<U>', '', line)
        line = re.sub('</U>', '', line)
        line = re.sub('\`', " ", line)
        line = re.sub('\-\-', ' ', line)
        line = re.sub('\. \. \.', ' . ', line)
        line = re.sub('\.  \.  \.', ' . ', line)
        line = re.sub(' +', ' ', line)
        line = line.lower()
        sentences[idx] = line
        embedded = embed_single(line, bpemb_en, seq_length)
        # print(embedded)
        embeddings.append(embedded)
    print('-> test mat shape:', embedded.shape)


    for i in range(len(sentences)):
        sentence = ''
        word = ''
        this_len = 0
        test_in_seq = np.expand_dims(embeddings[i], 0)
        # for h in range(seq_length):
        #     print(test_in_seq[0,:,:].shape)
        print('-> original sentence:')
        print(sentences[i])
        sent = bpemb_en.most_similar(test_in_seq[0,:,:])
        sent_str = ' '.join(word[0][1:] for word in sent)
        print('-> input words from embed:', sent_str)
        
        for j in range(30):
            this_len += 1
            arr = model.predict(test_in_seq)
            arr = np.expand_dims(arr, 0)

            test_in_seq = test_in_seq[:,1:,:]
            test_in_seq = np.concatenate((test_in_seq, arr), axis=1)
        sentence_sc = bpemb_en.most_similar(test_in_seq[0,:,:])
        sentence = ' '.join(word[0][1:] for word in sentence_sc)
            # print(arr)

        # sentence = re.sub(r"(.+?)\1+", r"\1", sentence)

        print(sentence)

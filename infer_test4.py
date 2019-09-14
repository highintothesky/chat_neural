# test the trained model v3
import re
import numpy as np
import keras
# import keras_self_attention
from keras_multi_head import MultiHead
from keras.utils import CustomObjectScope
from process_movie_lines3 import embed_single
from train_functional import root_mean_squared_error
from keras.models import load_model
from bpemb import BPEmb

from keras_self_attention import SeqSelfAttention
from gensim.models import FastText


if __name__ == '__main__':
    print('-> Loading att model')

    with CustomObjectScope({'SeqSelfAttention': SeqSelfAttention,
                            'MultiHead': MultiHead,
                            'root_mean_squared_error': root_mean_squared_error}):
        model = keras.models.load_model('models/func_9.h5')

    bpemb_en = BPEmb(lang="en", dim=100, vs=100000)
    seq_length = 30

    print(model.summary())
    sentences = ["hey, what's up?",
                 "kiss my ass",
                 "Are you really a robot? I can't believe it! now take this long string and do something with it",
                 "Gosh if only we could find Kat a boyfriend",
                 "you"]
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
        sentence_test = ''
        sentence_test_cos = ''
        # output_strings = []
        word = ''
        this_len = 0
        test_in_seq = np.expand_dims(embeddings[i], 0)
        seq_total = test_in_seq

        print('-> original sentence:', sentences[i])

        or_sent = ''
        for h in range(seq_length):
            if np.mean(np.abs(test_in_seq[0,h,:])) > 0.:
                sent = bpemb_en.most_similar(positive=[test_in_seq[0,h,:]])
                or_sent += sent[0][0][1:] + ' '
        print('-> reconstructed from embedding:', or_sent)

        for s_idx in range(1):
            for j in range(seq_length):
                this_len += 1
                arr = model.predict(test_in_seq)
                word = bpemb_en.most_similar(positive=arr)[0][0]
                if word[0] == '▁':
                    word = word[1:]
                sentence_test += word + ' '

                # word_cos = bpemb_en.most_similar_cosmul(positive=arr)[0][0]
                # if word_cos[0] == '▁':
                #     word_cos = word_cos[1:]
                # sentence_test_cos += word_cos + ' '

                arr = np.expand_dims(arr, 0)

                test_in_seq = test_in_seq[:,1:,:]
                test_in_seq = np.concatenate((test_in_seq, arr), axis=1)
                seq_total = np.concatenate((seq_total, arr), axis=1)

            out_arr = seq_total[0,seq_length:,:]
            sent_oba_sc = bpemb_en.most_similar(positive=out_arr)
            print('sent_oba_sc:', sent_oba_sc)

            sent_oba = ''
            for word_sc in sent_oba_sc:
                word = word_sc[0]
                if word[0] == '▁':
                    word = word[1:]
                sent_oba += word + ' '
            # sent_oba = ' '.join([])
            # print('-> concat test:', sentence_test)

        print('-> output sentence:', sentence_test)
        print('-> max scoring for first pred:', sent_oba)
        # print('-> output sentence cos:', sentence_test_cos)

        print(' ')
        # print('-> output second level:')
        # print(sentence2)

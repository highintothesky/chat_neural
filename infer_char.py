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

from keras_self_attention import SeqSelfAttention
from gensim.models import FastText


if __name__ == '__main__':
    print('-> Loading att model')

    with CustomObjectScope({'SeqSelfAttention': SeqSelfAttention,
                            'MultiHead': MultiHead,
                            'root_mean_squared_error': root_mean_squared_error}):
        model = keras.models.load_model('models/char_1.h5')

    seq_length = 100

    print(model.summary())
    sentences = ["hey, what's up?",
                 "kiss my ass",
                 "Are you really a robot? I can't believe it! now take this long string and do something with it",
                 "Gosh if only we could find Kat a boyfriend",
                 "you"]
    # embeddings = []
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
        # embedded = embed_single(line, bpemb_en, seq_length)
        # print(embedded)
        # embeddings.append(embedded)
    # print('-> test mat shape:', embedded.shape)


    for i in range(len(sentences)):
        filename = "data/lines_chars.txt"
        raw_text = open(filename, 'r', encoding='utf-8').read()
        raw_text = raw_text.lower()

        # create mapping of unique chars to integers
        chars = sorted(list(set(raw_text)))
        char_to_int = dict((c, i) for i, c in enumerate(chars))
        # summarize the loaded data
        n_chars = len(raw_text)
        n_vocab = len(chars)
        print("-> Total Characters: ", n_chars)
        print("-> Total Vocab: ", n_vocab)

        # we're not going to process all that
        raw_text = ' '.join([line for line in sentences])
        # prepare the dataset of input to output pairs encoded as integers
        seq_length = 100
        dataX = []
        dataY = []
        for i in range(0, n_chars - seq_length, 1):
        	seq_in = raw_text[i:i + seq_length]
        	seq_out = raw_text[i + seq_length]
        	dataX.append([char_to_int[char] for char in seq_in])
        	dataY.append(char_to_int[seq_out])

        n_patterns = len(dataX)

        # reshape X to be [samples, time steps, features]
        X = np.reshape(dataX, (n_patterns, seq_length, 1))
        # normalize
        X = X / float(n_vocab)
        # one hot encode the output variable
        y = np_utils.to_categorical(dataY)

        # pick a random seed
        # start = numpy.random.randint(0, len(dataX)-1)
        start = 0
        pattern = dataX[start]
        res = ' '
        print('pattern:', pattern)
        print( "Seed:")
        print( "\"", ''.join([int_to_char[value] for value in pattern]), "\"")
        # generate characters
        for i in range(100):
        	x = numpy.reshape(pattern, (1, len(pattern), 1))
        	x = x / float(n_vocab)
        	prediction = model.predict(x, verbose=0)
        	index = numpy.argmax(prediction)
        	result = int_to_char[index]
        	seq_in = [int_to_char[value] for value in pattern]
        	# sys.stdout.write(result)
            res += result + ' '
        	pattern.append(index)
        	pattern = pattern[1:len(pattern)]

        print('-> original sentence:', sentences[i])



        print('-> output sentence:', res)
        # print('-> max scoring for first pred:', sent_oba)
        # print('-> output sentence cos:', sentence_test_cos)

        print(' ')
        # print('-> output second level:')
        # print(sentence2)

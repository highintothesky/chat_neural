# process movie dialog for training the chatbot
# started using:
from bpemb import BPEmb
import numpy as np
import csv

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import h5py
from gensim.models import FastText
import re



def sentence_to_matrix(sentence, ft_model, seq_length):
    # for a single string
    sentence_list = sentence.split()
    emb_len = len(ft_model.wv['wtf'])

    total_mat = np.zeros((1, seq_length, emb_len))
    for i in range(len(sentence_list)):
        # we want to stick these words at the end of the vector
        this_idx = seq_length - len(sentence_list) + i
        word = sentence_list[i]
        total_mat[0,this_idx,:] = ft_model.wv[word]

    return total_mat

def sentences_to_matrix(sentence_list, ft_model):
    no_sentences = len(sentence_list)
    # seq_length = len(sentence_list[0].split())
    seq_length = 20
    emb_len = len(ft_model.wv['wtf'])
    res_arr = np.zeros((no_sentences, seq_length, emb_len))
    for idx, sent in enumerate(sentence_list):
        res_arr[idx,:,:] = sentence_to_matrix(sent, ft_model, seq_length)
    return res_arr

def encode_sentence_list(sentence_list, bpemb_en, seq_length):
    # process using BPE
    input_list = []
    output_list = []
    long_sentence = ''
    emb_len = bpemb_en.embed('test sentence').shape[1]

    for sentence in sentence_list:
        # print(sentence)
        long_sentence += sentence + ' '
        enc_mat = bpemb_en.embed(long_sentence)
        this_seq_length = enc_mat.shape[0]
        # we want long strings
        if this_seq_length > seq_length*2:
            long_sentence = ''
            # now loop over the dims of enc_mat
            for i in range(this_seq_length-seq_length):
                this_in_mat = enc_mat[i:i+seq_length,:]
                this_out_mat = enc_mat[i+seq_length,:]
                input_list.append(this_in_mat)
                output_list.append(this_out_mat)

    return np.array(input_list), np.array(output_list)

def embed_single(sentence, bpemb_en, seq_length):
    # embed single sentence (last has preference)
    enc_mat = bpemb_en.embed(sentence)
    this_seq_length = enc_mat.shape[0]
    emb_len = enc_mat.shape[1]
    if this_seq_length <= seq_length:
        arr = np.zeros((seq_length, emb_len))
        arr[-this_seq_length:,:] = enc_mat
    else:
        arr = enc_mat[-seq_length:,:]
    return arr

if __name__ == '__main__':
    # dict for storing all conversations
    conv_list = []
    conv_count = 0

    with open('data/movie_conversations.tsv') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        for row in reader:
            # print(row[3])
            cleaned = row[3].translate({ord(i): None for i in """[]'\""""}) # .strip("""[]'""").
            line_list = cleaned.split(' ')
            conv_list.append(line_list)
            # conv_dict[conv_count]['line_list'] = line_list
            conv_count += 1

    # now let's read lines, put them in a dict
    line_dict = {}
    with open('data/movie_lines_noquotes.tsv') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        for row in reader:
            # print(row[3])
            line_name = row[0].strip('"')
            # print(row)
            try:
                line = row[4]
            except Exception as ex:
                line_raw = row[3].replace('\x85', '\t')
                line_new = line_raw.split('\t')
                if len(line_new) > 1:
                    line = line_new[1]
                else:
                    # print(line_raw)
                    # print(ex)
                    line = "..."
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

            line_dict[line_name] = line

    # overwrite because why not
    write_txt = False
    encode = True
    if write_txt:
        with open('data/lines_raw.txt', 'w') as f:
            for line_list in conv_list:
                for line_id in line_list:
                    f.write(line_dict[line_id] + ' \n')

    # store the accepted convos
    # at least x percent of the lines must have this length
    # min_sent_len = 40
    # len_perc_max = 0.5
    in_sentences = []
    # if write_txt:
    #     f_in = open('data/input_sentences.txt', 'w')
    #     f_out = open('data/output_words.txt', 'w')

    for line_list in conv_list:
        # 1 because we're going to add a stop sign later
        total_len = 1
        sentences = []
        for line_id in line_list:
            in_sentences.append(line_dict[line_id])


    print('-> Sentences accepted:')
    print(len(in_sentences))


    if encode:
        print('-> loading BPE wordembed model')
        bpemb_en = BPEmb(lang="en", dim=100, vs=100000)
        seq_length = 30
        print('-> processing lines to embed')
        # let's only process the first ... idk
        oba_inputs, oba_outputs = encode_sentence_list(in_sentences[300000:], bpemb_en, seq_length)
        print('-> Writing to file')


        h5f = h5py.File('data/processed_bpe_test.h5', 'w')
        h5f.create_dataset('input', data=oba_inputs)
        h5f.create_dataset('output', data=oba_outputs)
        print('-> wrote arrays to hdf5')

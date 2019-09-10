# process movie dialog for training the chatbot
import numpy as np
import csv

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import h5py
from gensim.models import FastText
import re

def sentences_to_matrix(sentence_list, ft_model, max_sen_len):
    emb_len = len(ft_model.wv['wtf'])
    no_sentences = len(sentence_list)
    sent_idx = 0
    total_mat = np.zeros((no_sentences, max_sen_len, emb_len))
    for sent in sentence_list:
        this_list = sent.split()
        # vec_list = []
        word_idx = 0
        sent_mat = np.zeros((max_sen_len, emb_len))

        for word in this_list:
            sent_mat[word_idx, :] = ft_model.wv[word]
            word_idx += 1
            if word_idx == max_sen_len:
                break

        total_mat[sent_idx, :, :] = sent_mat
        sent_idx += 1

    return total_mat


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
            line = re.sub('\.\.\.', ' ', line)
            line = re.sub('\. \. \.', '. ', line)
            line = re.sub('\?\.', '?', line)

            line_dict[line_name] = line

    # overwrite because why not
    write_raw = True
    if write_raw:
        with open('data/lines_raw.txt', 'w') as f:
            for line_list in conv_list:
                for line_id in line_list:
                    f.write('sos ' + line_dict[line_id] + ' eos \n')

    # store the accepted convos
    # at least x percent of the lines must have this length
    min_sent_len = 6
    len_perc_max = 0.2
    conv_accepted = []
    sent_accepted = []
    utterances = []
    responses = []
    f_utt = open('data/utterances.txt', 'w')
    f_resp = open('data/responses.txt', 'w')

    for line_list in conv_list:
        too_short = 0.
        for line_id in line_list:
            if len(line_dict[line_id].split()) < min_sent_len:
                too_short += 1
        if too_short/len(line_list) < len_perc_max:
            conv_accepted.append(line_list)
            for idx, line_id in enumerate(line_list):
                sent_accepted.append('sos ' + line_dict[line_id] + ' eos')
                if idx < len(line_list) - 1:
                    utt = 'sos ' + line_dict[line_id] + ' eos'
                    utterances.append(utt)
                    f_utt.write(utt + '\n')
                    next_id = line_list[idx+1]
                    resp = 'sos ' + line_dict[next_id] + ' eos'
                    responses.append(resp)
                    f_resp.write(resp + '\n')


    print('-> convo length before cull:')
    print(len(conv_list))
    print('-> convo length after cull:')
    print(len(conv_accepted))

    print('-> utterances:')
    print(len(utterances))
    print('-> responses:')
    print(len(utterances))

    ft = True
    if ft:
        print('-> loading fasttext wordembed model')
        ft_model = FastText.load('models/fasttext2')
        max_sen_len = 20
        # sent_mat = sentences_to_matrix(sent_accepted)
        oba_utt = sentences_to_matrix(utterances, ft_model, max_sen_len)
        oba_resp = sentences_to_matrix(responses, ft_model, max_sen_len)
        print(utterances[0:3])
        print(oba_utt[0:3,:,:])

        h5f = h5py.File('data/processed1.h5', 'w')
        h5f.create_dataset('utterances', data=oba_utt)
        h5f.create_dataset('responses', data=oba_resp)
        print('-> wrote arrays to hdf5')

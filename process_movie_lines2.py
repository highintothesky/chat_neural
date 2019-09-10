# process movie dialog for training the chatbot
# now made for producing the next word
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
    # seq_length = len(sentence_list)
    # no_sentences = len(sentence_list)
    # sent_idx = 0
    total_mat = np.zeros((1, seq_length, emb_len))
    for i in range(len(sentence_list)):
        word = sentence_list[i]
        total_mat[0,i,:] = ft_model.wv[word]

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
    write_raw = False
    if write_raw:
        with open('data/lines_raw.txt', 'w') as f:
            for line_list in conv_list:
                for line_id in line_list:
                    f.write('sos ' + line_dict[line_id] + ' eos \n')

    # store the accepted convos
    # at least x percent of the lines must have this length
    min_sent_len = 20
    len_perc_max = 0.5
    in_sentences = []
    out_words = []
    f_in = open('data/input_sentences.txt', 'w')
    f_out = open('data/output_words.txt', 'w')

    for line_list in conv_list:
        # 1 because we're going to add a stop sign later
        total_len = 1
        sentences = []
        for line_id in line_list:
            this_line = line_dict[line_id].split()
            total_len += len(this_line)
            sentences += this_line
        sentences.append('<eos>')

        if total_len > min_sent_len:
            for i in range(0, total_len - min_sent_len - 1):
                input_list = sentences[i:i+min_sent_len]
                input_str = ' '.join([word for word in input_list])
                output_str = sentences[i+min_sent_len+1]
                in_sentences.append(input_str)
                out_words.append(output_str)
                f_in.write(input_str + '\n')
                f_out.write(output_str + '\n')


    print('-> Sentences accepted:')
    print(len(in_sentences))

    ft = True
    if ft:
        print('-> loading fasttext wordembed model')
        ft_model = FastText.load('models/fasttext2')
        # max_sen_len = 20
        oba_inputs = sentences_to_matrix(in_sentences[:500000], ft_model)
        oba_outputs = sentences_to_matrix(out_words[:500000], ft_model)
        # normalize, print max, min
        # first add abs(min)
        # then decide max
        # then divide by max
        # dat_min = np.min(oba_utt)
        # print('-> Lowest value in data:')
        # print(dat_min)
        # oba_utt += np.abs(dat_min)
        # oba_resp += np.abs(dat_min)
        # dat_max = np.max(oba_utt)
        # print('-> highest value in data:')
        # print(dat_max)
        # oba_utt = np.divide(oba_utt, dat_max)
        # oba_resp = np.divide(oba_resp, dat_max)
        # print('-> max:')
        # print(np.max(oba_inputs))
        # print('-> min:')
        # print(np.min(oba_inputs))

        h5f = h5py.File('data/processed2.h5', 'w')
        h5f.create_dataset('input', data=oba_inputs)
        h5f.create_dataset('output', data=oba_outputs)
        print('-> wrote arrays to hdf5')

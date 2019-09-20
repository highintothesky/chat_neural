# process chars, put in h5
import h5py
import re
import numpy as np

def main():
    filename = "data/mixed.txt"
    raw_text = open(filename, 'r', encoding='utf-8').read()
    raw_text = raw_text.lower()
    raw_text = re.sub('\n', " ", raw_text)

    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    char_list = list(char_to_int.keys())
    raw_text = ''.join([i for i in raw_text if i in char_list])

    n_chars_total = len(raw_text)
    n_vocab = len(char_list)
    print(char_to_int)

    seq_len = 100
    start_idx = 0
    chunk_size = 1000000
    end_idx = start_idx + chunk_size

    tot_chunks = int((n_chars_total - seq_len)/chunk_size) + 1
    for j in range(tot_chunks):
        text_chunk = raw_text[start_idx:end_idx]
        n_chars = len(text_chunk)

        start_idx = end_idx
        if j == tot_chunks -1:
            # last text chunk
            end_idx = -1
        else:
            end_idx += chunk_size

        # now we cut it up into sequences
        sentences = []
        next_chars = []
        for i in range(0, n_chars - seq_len, 1):
            sentences.append(text_chunk[i:i+seq_len])
            next_chars.append(text_chunk[i+seq_len])

        n_patterns = len(sentences)
        print("Total Patterns: ", n_patterns)

        X = np.zeros((n_patterns, seq_len, n_vocab), dtype=np.bool)
        y = np.zeros((n_patterns, n_vocab), dtype=np.bool)

        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X[i, t, char_to_int[char]] = 1
            y[i, char_to_int[next_chars[i]]] = 1

        print('- Input:',X.shape)
        print('- Output:',y.shape)

        # save the array
        h5f = h5py.File('data/processed{}.h5'.format(j), 'w')
        h5f.create_dataset('input', data=X)
        h5f.create_dataset('output', data=y)
        h5f.close()


if __name__ == '__main__':
    main()

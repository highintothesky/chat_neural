# clean a txt

char_to_int = {' ': 0, '!': 1, '"': 2, "'": 3, ',': 4, '-': 5, '.': 6, '/': 7, '0': 8, '1': 9, '2': 10, '3': 11, '4': 12, '5': 13, '6': 14, '7': 15, '8': 16, '9': 17, ':': 18, ';': 19, '<': 20, '>': 21, '?': 22, '@': 23, 'a': 24, 'b': 25, 'c': 26, 'd': 27, 'e': 28, 'f': 29, 'g': 30, 'h': 31, 'i': 32, 'j': 33, 'k': 34, 'l': 35, 'm': 36, 'n': 37, 'o': 38, 'p': 39, 'q': 40, 'r': 41, 's': 42, 't': 43, 'u': 44, 'v': 45, 'w': 46, 'x': 47, 'y': 48, 'z': 49, '~': 50}
char_list = list(char_to_int.keys())
char_list.append('\n')

raw_text = open('data/subreddit_comments.txt', 'r', encoding='utf-8').read().lower()
out_file = open('data/subreddit_comments_cleaned.txt', 'w')

raw_text = ''.join([i for i in raw_text if i in char_list])

out_file.write(raw_text)

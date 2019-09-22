# process sarcastic csv
import csv


def main():
    with open('data/train-balanced-sarcasm.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        char_to_int = {' ': 0, '!': 1, '"': 2, "'": 3, ',': 4, '-': 5, '.': 6, '/': 7, '0': 8, '1': 9, '2': 10, '3': 11, '4': 12, '5': 13, '6': 14, '7': 15, '8': 16, '9': 17, ':': 18, ';': 19, '<': 20, '>': 21, '?': 22, '@': 23, 'a': 24, 'b': 25, 'c': 26, 'd': 27, 'e': 28, 'f': 29, 'g': 30, 'h': 31, 'i': 32, 'j': 33, 'k': 34, 'l': 35, 'm': 36, 'n': 37, 'o': 38, 'p': 39, 'q': 40, 'r': 41, 's': 42, 't': 43, 'u': 44, 'v': 45, 'w': 46, 'x': 47, 'y': 48, 'z': 49, '~': 50}
        char_list = list(char_to_int.keys())
        print(char_list)

        f_out = open('data/sarcasm_convo.txt', 'w')

        accepted_parent = []
        accepted_reaction = []
        for row in csv_reader:
            if row[0] == '1':
                # print(row[1])
                parent = row[-1].lower()
                reaction = row[1].lower()
                # remove unwanted chars
                parent = ''.join([c for c in parent if c in char_list])
                reaction = ''.join([c for c in reaction if c in char_list])
                f_out.write(parent + '\n')
                f_out.write(reaction + '\n')
                accepted_parent.append(parent)
                accepted_reaction.append(reaction)

        f_out.close()
        print('-> sarcastic comments:', len(accepted_reaction))
        for i in range(10):
            print(accepted_parent[i])
            print(accepted_reaction[i])
            print('\n')

if __name__ == '__main__':
    main()

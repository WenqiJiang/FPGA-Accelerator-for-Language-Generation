import csv

def get_dict():
    """
    return the index2word dictionary,
    notice that even indexes are represented as string
    """
    idx_word = dict()
    with open('idx_word.csv', mode='r') as infile:
        csv_reader = csv.reader(infile)
        idx_word = {rows[0]:rows[1] for rows in csv_reader}

    return idx_word

def get_sequence(file_name):
    """
    given the file name, return the array containing the index sequence,
    notice that the indexes are represented as string
    """
    with open(file_name, mode='r') as f:
        sequence = f.read()

    return str.split(sequence, "\t")

def interpret_sequence(idx_sequence, idx_word_dict):
    """
    given an index sequence, and the dictionary of index to word,
    return the array of interpreted sequence which consists of words
    """
    word_array = []
    for idx in idx_sequence:
        if idx in idx_word_dict:
            word_array.append(idx_word_dict[idx])
        else:
            word_array.append('UNK')

    return word_array

def print_word_sequence(word_sequence):
    """ as the function name """
    for word in word_sequence:
        print(word, end =" ")
    print("\n")

if __name__ == "__main__":

    file_name = "idx_sequence.txt"
    idx_word = get_dict()
    idx_sequence = get_sequence(file_name)
    word_sequence = interpret_sequence(idx_sequence, idx_word)
    print_word_sequence(word_sequence)

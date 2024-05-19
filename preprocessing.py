# preprocess.py
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import get_file
from collections import Counter
import nltk
from nltk import word_tokenize
import os

nltk.download('punkt')

# Load and preprocess the text data
def preprocess_data():
    url = 'https://www.gutenberg.org/cache/epub/100/pg100.txt'
    path = get_file('pg100.txt', origin=url)
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()

    text = []
    start = False
    for line in lines:
        line = line.strip().lower()
        if "*** START OF THE PROJECT GUTENBERG EBOOK THE COMPLETE WORKS OF WILLIAM SHAKESPEARE ***".lower() in line and not start:
            start = True
        if "*** END OF THE PROJECT GUTENBERG EBOOK THE COMPLETE WORKS OF WILLIAM SHAKESPEARE ***".lower() in line:
            break
        if not start or len(line) == 0:
            continue
        text.append(line)

    text = " ".join(text)
    voc_chars = sorted(set(text))
    char_indices = {c: i for i, c in enumerate(voc_chars)}
    indices_char = {i: c for i, c in enumerate(voc_chars)}

    tokens = word_tokenize(text)
    freq = Counter(tokens)
    ordered_word_list = freq.most_common()

    rank_counts = np.array([[rank + 1, count] for rank, (_, count) in enumerate(ordered_word_list)])
    plt.figure(figsize=(20, 5))
    plt.title('Word counts versus rank')
    plt.scatter(rank_counts[:, 0], rank_counts[:, 1])
    plt.yscale('log')
    plt.show()

    print(f'Vocabulary size: {len(freq)}')
    for i in range(1000, len(freq), 1000):
        print(f'{i} : {np.sum(rank_counts[:i, 1]) / np.sum(rank_counts[:, 1]):.2f}')

    maximum_seq_length = 30
    time_step = 4
    sentences = []
    next_char = []
    n = len(text)
    for i in range(0, n - maximum_seq_length, time_step):
        sentences.append(text[i:i + maximum_seq_length])
        next_char.append(text[i + maximum_seq_length])
    
    print(f'Number of Sequences: {len(sentences)}')

    X = np.zeros((len(sentences), maximum_seq_length, len(voc_chars)), dtype=bool)
    y = np.zeros((len(sentences), len(voc_chars)), dtype=bool)
    
    for i, sentence in enumerate(sentences):
        for j, char in enumerate(sentence):
            X[i, j, char_indices[char]] = 1
        y[i, char_indices[next_char[i]]] = 1

    return X, y, maximum_seq_length, voc_chars, char_indices, indices_char

X, y, maximum_seq_length, voc_chars, char_indices, indices_char = preprocess_data()
np.savez('preprocessed_data.npz', X=X, y=y, maximum_seq_length=maximum_seq_length, 
         voc_chars=voc_chars, char_indices=char_indices, indices_char=indices_char)

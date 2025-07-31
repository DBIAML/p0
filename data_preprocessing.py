# data_preprocessing.py

from tensorflow.keras.preprocessing.sequence import pad_sequences


def build_vocab(data):
    # On construit les vocabulaires pour les mots et les tags
    word_set = set()
    tag_set = set()
    for words, tags in data:
        word_set.update(words)
        tag_set.update(tags)
    word2idx = {w: i + 2 for i, w in enumerate(sorted(word_set))}
    word2idx['PAD'] = 0  # Pour le padding
    word2idx['UNK'] = 1  # Pour les mots inconnus
    tag2idx = {t: i for i, t in enumerate(sorted(tag_set))}
    return word2idx, tag2idx


def encode_data(data, word2idx, tag2idx, max_len=20):
    # On encode les mots et les tags en indices numériques
    X = [
        [word2idx.get(w, word2idx['UNK']) for w in words] 
        for words, tags in data
    ]
    y = [[tag2idx[t] for t in tags] for words, tags in data]
    # On applique le padding pour avoir des séquences de même longueur
    X = pad_sequences(X, maxlen=max_len, padding='post', value=word2idx['PAD'])
    y = pad_sequences(y, maxlen=max_len, padding='post', value=tag2idx['O'])
    return X, y

    
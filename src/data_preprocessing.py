# data_preprocessing.py

from tensorflow.keras.preprocessing.sequence import pad_sequences


def build_vocab(data):
    """
    Construit les vocabulaires pour les mots et les tags à partir des données annotées.

    Parcourt la liste de tuples (mots, tags) pour extraire tous les mots et tags uniques,
    puis crée deux dictionnaires de correspondance mot->indice et tag->indice.
    Le dictionnaire des mots inclut les tokens spéciaux 'PAD' (pour le padding) et 'UNK' (pour les mots inconnus).

    Args:
        data (list of tuple): Liste de tuples, chaque tuple contenant une liste de mots et une liste de tags.

    Returns:
        tuple:
            - word2idx (dict): Dictionnaire associant chaque mot unique à un indice entier.
            - tag2idx (dict): Dictionnaire associant chaque tag unique à un indice entier.
    """
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
    """
    Encode les séquences de mots et de tags en indices numériques avec padding.

    Transforme chaque mot et tag en son indice correspondant à l'aide des dictionnaires fournis,
    puis applique un padding pour obtenir des séquences de longueur fixe.

    Args:
        data (list of tuple): Liste de tuples (mots, tags) à encoder.
        word2idx (dict): Dictionnaire mot->indice.
        tag2idx (dict): Dictionnaire tag->indice.
        max_len (int, optional): Longueur maximale des séquences de sortie. Par défaut à 20.

    Returns:
        tuple:
            - X (numpy.ndarray): Tableau des séquences de mots encodés, paddées à max_len.
            - y (numpy.ndarray): Tableau des séquences de tags encodés, paddées à max_len.
    """
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

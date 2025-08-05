import unittest
import numpy as np
from data_preprocessing import build_vocab, encode_data

class TestDataPreprocessing(unittest.TestCase):
    """
    Classe de tests unitaires pour les fonctions du module data_preprocessing.py.
    Teste la construction des vocabulaires et l'encodage des données avec padding.
    """


def setUp(self):
    """
    Initialise des exemples de données pour les tests.
    """
    self.data = [
    (['Jean', 'aime', 'Paris'], ['B-PER', 'O', 'B-LOC']),
    (['Marie', 'visite', 'Londres'], ['B-PER', 'O', 'B-LOC']),
    (['Pierre', 'mange', 'une', 'pomme'], ['B-PER', 'O', 'O', 'O'])
    ]
    self.max_len = 5


def test_build_vocab(self):
    """
    Vérifie que build_vocab construit correctement les dictionnaires de mots et de tags.
    """
    word2idx, tag2idx = build_vocab(self.data)
    # Vérifie la présence des tokens spéciaux
    self.assertIn('PAD', word2idx)
    self.assertIn('UNK', word2idx)
    # Vérifie que tous les mots et tags uniques sont présents
    all_words = {w for sent, _ in self.data for w in sent}
    all_tags = {t for _, tags in self.data for t in tags}
    for w in all_words:
        self.assertIn(w, word2idx)
    for t in all_tags:
        self.assertIn(t, tag2idx)


def test_encode_data_shape_and_padding(self):
    """
    Vérifie que encode_data retourne des tableaux de la bonne forme
    et que le padding est appliqué correctement.
    """
    word2idx, tag2idx = build_vocab(self.data)
    X, y = encode_data(self.data, word2idx, tag2idx, max_len=self.max_len)
    # Vérifie la forme des tableaux
    self.assertEqual(X.shape, (3, self.max_len))
    self.assertEqual(y.shape, (3, self.max_len))
    # Vérifie que les séquences sont bien paddées à droite
    for i, (sent, tags) in enumerate(self.data):
        self.assertTrue(np.all(X[i, len(sent):] == word2idx['PAD']))
        self.assertTrue(np.all(y[i, len(tags):] == tag2idx['O']))


def test_encode_data_unknown_word(self):
    """
    Vérifie que les mots inconnus sont bien encodés avec l'indice 'UNK'.
    """
    word2idx, tag2idx = build_vocab(self.data)
    # Ajoute une phrase avec un mot inconnu
    test_data = [(['Inconnu', 'aime', 'Paris'], ['O', 'O', 'B-LOC'])]
    X, y = encode_data(test_data, word2idx, tag2idx, max_len=self.max_len)
    self.assertEqual(X[0, 0], word2idx['UNK'])

if __name__ == '__main__':
    unittest.main()

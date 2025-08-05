import unittest
from tensorflow.keras import Model
from model import build_model

class TestBuildModel(unittest.TestCase):
    """
    Classe de tests unitaires pour la fonction build_model du module model.py.

    Cette classe vérifie que la fonction build_model retourne bien un modèle Keras compilé
    avec la bonne architecture et les bons paramètres.

    Méthodes
    --------
    test_build_model_structure()
    Vérifie la structure, les dimensions d'entrée et de sortie, et la compilation du modèle.
    """


def test_build_model_structure(self):
    """
    Teste la fonction build_model pour s'assurer que :
        - Le modèle retourné est une instance de tensorflow.keras.Model.
        - Les dimensions d'entrée et de sortie sont correctes.
        - Le modèle est compilé avec les bons paramètres.
        - Le nombre de couches principales correspond à l'architecture attendue.

    Args:
        Aucun argument n'est requis pour ce test.

    Returns:
        None. Les assertions lèvent une erreur en cas d'échec.
    """
    input_dim = 100 # taille du vocabulaire
    output_dim = 5 # nombre de classes de sortie
    input_length = 10 # longueur maximale des séquences

    model = build_model(input_dim, output_dim, input_length)

    # Vérifie que le modèle est une instance de keras.Model
    self.assertIsInstance(model, Model)

    # Vérifie la forme de l'entrée
    self.assertEqual(model.input_shape, (None, input_length))

    # Vérifie la forme de la sortie
    self.assertEqual(model.output_shape, (None, input_length, output_dim))

    # Vérifie que le modèle est compilé avec la bonne fonction de perte
    self.assertEqual(model.loss, 'sparse_categorical_crossentropy')

    # Vérifie la présence des couches principales attendues
    layer_types = [type(layer).__name__ for layer in model.layers]
    self.assertIn('Embedding', layer_types)
    self.assertIn('Bidirectional', layer_types)
    self.assertIn('TimeDistributed', layer_types)

if __name__ == '__main__':
    unittest.main()

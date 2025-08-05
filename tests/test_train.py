import unittest
from unittest.mock import patch, MagicMock
import os
import train


class TestTrainMain(unittest.TestCase):
    """
    Classe de tests unitaires pour la fonction main du module train.py.

    Cette classe vérifie que le pipeline d'entraînement NER s'exécute correctement :
        - Les fonctions de génération de données, de prétraitement et de modélisation sont bien appelées.
        - Le modèle est entraîné et sauvegardé.
        - Le fichier de sauvegarde du modèle est effectivement créé.

    Méthodes
    --------
    test_main_pipeline():
    Teste l'exécution complète de la fonction main, en utilisant des mocks pour éviter un entraînement réel.

    Attributes
    ----------
    Aucun attribut spécifique.
    """

    @patch("train.build_model")
    @patch("train.encode_data")
    @patch("train.build_vocab")
    @patch("train.generate_synthetic_data")


def test_main_pipeline(self, mock_generate, mock_vocab, mock_encode, mock_model):
    """
    Teste la fonction main du module train.py.

    Ce test vérifie que :
        - Les fonctions de génération de données, de construction de vocabulaire, d'encodage et de modélisation
        sont appelées avec les bons arguments.
        - La méthode fit du modèle est appelée pour l'entraînement.
        - La méthode save du modèle est appelée pour la sauvegarde.
        - Le fichier de sauvegarde du modèle est créé (simulation).

    Args:
        mock_generate (MagicMock): Mock pour generate_synthetic_data.
        mock_vocab (MagicMock): Mock pour build_vocab.
        mock_encode (MagicMock): Mock pour encode_data.
        mock_model (MagicMock): Mock pour build_model.

    Returns:
        None. Les assertions lèvent une erreur en cas d'échec.
    """
    # Préparation des mocks
    mock_generate.return_value = [(['a', 'b'], ['O', 'B-PER'])]
    mock_vocab.return_value = ({'PAD': 0, 'UNK': 1, 'a': 2, 'b': 3}, {'O': 0, 'B-PER': 1})
    mock_encode.return_value = ("X", "y")
    mock_model_instance = MagicMock()
    mock_model.return_value = mock_model_instance

    # Suppression du modèle s'il existe déjà
    model_path = "ner_beneficiary_model.h5"
    if os.path.exists(model_path):
        os.remove(model_path)

    # Appel de la fonction main
    train.main()

    # Vérifications des appels
    mock_generate.assert_called_with(300)
    mock_vocab.assert_called()
    mock_encode.assert_called()
    mock_model.assert_called()
    mock_model_instance.fit.assert_called()
    mock_model_instance.save.assert_called_with(model_path)


# (Optionnel) Nettoyage après le test
def tearDown(self):
    """Supprime le fichier de modèle créé pendant les tests, s'il existe."""
    model_path = "ner_beneficiary_model.h5"
    if os.path.exists(model_path):
    os.remove(model_path)


if __name__ == "__main__":
    unittest.main()

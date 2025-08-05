# train.py

from data_generator import generate_synthetic_data
from data_preprocessing import build_vocab, encode_data
from model import build_model


def main():
    """
    Exécute le pipeline complet d'entraînement du modèle de reconnaissance d'entités nommées (NER).

    Cette fonction effectue les étapes suivantes :
        1. Génère des données synthétiques pour l'entraînement.
        2. Construit les vocabulaires de mots et de tags à partir des données.
        3. Encode les données sous forme de séquences d'indices.
        4. Construit et compile le modèle NER.
        5. Entraîne le modèle sur les données encodées avec une validation croisée.
        6. Sauvegarde le modèle entraîné au format HDF5.

    Aucune valeur de retour.
    """
    # Génération de données
    data = generate_synthetic_data(300)
    # Construction des vocabulaires
    word2idx, tag2idx = build_vocab(data)
    max_len = 20
    # Encodage des données
    X, y = encode_data(data, word2idx, tag2idx, max_len)
    # Construction du modèle
    model = build_model(
        input_dim=len(word2idx),
        output_dim=len(tag2idx),
        input_length=max_len
    )
    # Entraînement
    model.fit(
        X, y, batch_size=8, epochs=3, validation_split=0.2
    )
    # Sauvegarde du modèle
    model.save("ner_beneficiary_model.h5")


if __name__ == "__main__":
    main()

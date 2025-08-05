# model.py

from tensorflow.keras import layers, models


def build_model(input_dim, output_dim, input_length):
    """
    Construit et compile un modèle de reconnaissance de séquence avec Keras.

    Le modèle est composé d'une couche d'embedding, d'une couche LSTM bidirectionnelle
    et d'une couche TimeDistributed avec une sortie softmax pour l'étiquetage de séquence
    (par exemple, pour la reconnaissance d'entités nommées).

    Args:
        input_dim (int): Taille du vocabulaire (nombre de mots uniques).
        output_dim (int): Nombre de classes de sortie (nombre de tags uniques).
        input_length (int): Longueur maximale des séquences en entrée.

    Returns:
        tensorflow.keras.Model: Modèle Keras compilé prêt à l'entraînement.
    """
    inputs = layers.Input(shape=(input_length,))
    x = layers.Embedding(
        input_dim=input_dim, output_dim=64, mask_zero=True
    )(inputs)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
    outputs = layers.TimeDistributed(
        layers.Dense(output_dim, activation='softmax')
    )(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

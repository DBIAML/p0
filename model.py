# model.py

from tensorflow.keras import layers, models


def build_model(input_dim, output_dim, input_length):
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


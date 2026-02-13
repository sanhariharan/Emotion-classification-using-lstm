import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

def build_model(vocab_size, max_len, embed_dim=128):

    model = Sequential([
        Embedding(vocab_size, embed_dim, input_length=max_len),

        Bidirectional(LSTM(128, return_sequences=False)),

        Dropout(0.5),

        Dense(64, activation='relu'),

        Dense(6, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

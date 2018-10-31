import numpy as np

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, TimeDistributed, Activation
from keras.models import Model
from keras.preprocessing import sequence

# config
use_dropout = True

def LSTM_Model(vocab_size, embedding_size, hidden_size, n_classes, num_steps):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size))
    model.add(LSTM(hidden_size, return_sequences=True, stateful=False))
    if use_dropout:
        model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(n_classes)))
    model.add(Activation('softmax'))

    model.summary()

    return model

def BiLSTM_Model(vocab_size, embedding_size, hidden_size, n_classes, num_steps):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size))
    model.add(Bidirectional(LSTM(hidden_size, return_sequences=True, stateful=False)))
    if use_dropout:
        model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(n_classes)))
    model.add(Activation('softmax'))

    model.summary()

    return model

if __name__ == "__main__":
    BiLSTM_Model(10000, 256, 128, 3, 20)
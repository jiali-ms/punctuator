import numpy as np

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, TimeDistributed, Activation, Convolution1D, MaxPool1D
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

def LSTM2Layer_Model(vocab_size, embedding_size, hidden_size, n_classes, num_steps):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size))
    model.add(LSTM(hidden_size, return_sequences=True, stateful=False))
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

def CLSTM(vocab_size, embedding_size, hidden_size, n_classes, num_steps):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size))
    model.add(Convolution1D(128, 3, padding='same', strides=1))
    model.add(Activation('relu'))
    model.add(LSTM(hidden_size, return_sequences=True, stateful=False))
    if use_dropout:
        model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(n_classes)))
    model.add(Activation('softmax'))

    model.summary()

    return model

def CBiLSTM(vocab_size, embedding_size, hidden_size, n_classes, num_steps):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size))
    model.add(Convolution1D(128, 3, padding='same', strides=1))
    model.add(Activation('relu'))
    model.add(Bidirectional(LSTM(hidden_size, return_sequences=True, stateful=False)))
    if use_dropout:
        model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(n_classes)))
    model.add(Activation('softmax'))

    model.summary()

    return model


if __name__ == "__main__":
    CBiLSTM(10000, 256, 128, 3, 20)
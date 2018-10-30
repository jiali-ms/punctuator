from keras.models import Sequential

from data import CharVocab, Corpus
from util import data_generator
from model import LSTM_Model, BiLSTM_Model

from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", "-bs", type=int, default=256, help="batch size")
parser.add_argument("--step_size", "-ts", type=int, default=20, help="step size")
parser.add_argument("--embedding_size", "-es", type=int, default=256, help="embedding size")
parser.add_argument("--hidden_size", "-hs", type=int, default=128, help="hidden size")
args = parser.parse_args()

# specify GPU
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# load corpus and vocab
vocab = CharVocab(100000) # 100k
corpus = Corpus(vocab, debug=True)
output_punc = {0:vocab.decode(0), 1:vocab.decode(1), 2:vocab.decode(2)}

# train with keras
checkpoint = ModelCheckpoint('weight/model_{epoch:02d}_{val_loss:02f}.hdf5',
                             verbose=1, save_best_only=True, mode='auto')

earlystop = EarlyStopping(patience=1)

model = LSTM_Model(len(vocab), args.embedding_size, args.hidden_size, len(output_punc), args.step_size)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

model.fit_generator(
    generator=data_generator(corpus.encoded_train, args.batch_size, args.step_size, len(output_punc)),
    validation_data=data_generator(corpus.encoded_dev, args.batch_size, args.step_size, len(output_punc)),
    steps_per_epoch=len(corpus.encoded_train[0])//(args.batch_size*args.step_size),
    validation_steps=len(corpus.encoded_dev[0])//(args.batch_size*args.step_size),
    epochs=1,
    callbacks=[checkpoint, earlystop],
    shuffle=False) # We will use stateful LSTM, don't shuffle. Also, data is already shuffled before.

model.evaluate_generator(data_generator(corpus.encoded_test, args.batch_size, args.step_size, len(output_punc)),
                         steps=len(corpus.encoded_test[0])//(args.batch_size*args.step_size))
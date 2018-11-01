import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", "-bs", type=int, default=512, help="batch size")
parser.add_argument("--step_size", "-ts", type=int, default=40, help="step size")
parser.add_argument("--embedding_size", "-es", type=int, default=256, help="embedding size")
parser.add_argument("--hidden_size", "-hs", type=int, default=128, help="hidden size")
parser.add_argument("--model", "-m", type=str, default='lstm', help="type model name among 'lstm', 'bilstm'")
parser.add_argument("--gpu", "-g", type=int, default=2, help="which gpu to use")
parser.add_argument("--class_weight", "-cw", type=int, default=50, help="class weight for comma and period")
args = parser.parse_args()

from keras.models import Sequential

from data import CharVocab, Corpus
from util import data_generator
from model import LSTM_Model, BiLSTM_Model, LSTM2Layer_Model, CLSTM, CBiLSTM

from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score

models ={'lstm': LSTM_Model,
         'bilstm': BiLSTM_Model,
         '2lstm': LSTM2Layer_Model,
         'clstm': CLSTM,
         'cbilstm': CBiLSTM}

# specify GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# load corpus and vocab
vocab = CharVocab(100000) # 100k
corpus = Corpus(vocab, debug=False)
output_punc = {0:vocab.decode(0), 1:vocab.decode(1), 2:vocab.decode(2)}

# train with keras
checkpoint = ModelCheckpoint('weight/model_%s_{epoch:02d}_{val_loss:02f}.hdf5' % args.model,
                             verbose=1, save_best_only=True, mode='auto')

earlystop = EarlyStopping(patience=1)


print('%s model is used' % args.model)
model = models[args.model](len(vocab), args.embedding_size, args.hidden_size, len(output_punc), args.step_size)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

model.fit_generator(
    generator=data_generator(corpus.encoded_train, args.batch_size, args.step_size, len(output_punc)),
    validation_data=data_generator(corpus.encoded_dev, args.batch_size, args.step_size, len(output_punc)),
    steps_per_epoch=len(corpus.encoded_train[0])//(args.batch_size*args.step_size),
    validation_steps=len(corpus.encoded_dev[0])//(args.batch_size*args.step_size),
    class_weight=[1, args.class_weight, args.class_weight],  # blank comma period
    epochs=10,
    callbacks=[checkpoint, earlystop],
    shuffle=False) # We will use stateful LSTM, don't shuffle. Also, data is already shuffled before.

model.evaluate_generator(data_generator(corpus.encoded_test, args.batch_size, args.step_size, len(output_punc)),
                         steps=len(corpus.encoded_test[0])//(args.batch_size*args.step_size))
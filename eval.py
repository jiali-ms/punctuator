import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default='model_bilstm_10_0.086493.hdf5', help="the model to use in evaluation")
parser.add_argument("--batch_size", "-bs", type=int, default=512, help="batch size")
parser.add_argument("--step_size", "-ts", type=int, default=40, help="step size")
parser.add_argument("--merge_punc", "-mp", action='store_true', help="merge all punctuation in evaluation")
args = parser.parse_args()

import os
from keras.models import load_model
from util import data_generator, generator_y_true
from data import CharVocab, Corpus
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
import numpy as np

model = load_model(os.path.join('weight', args.model))

# load corpus and vocab
vocab = CharVocab(100000) # 100k
corpus = Corpus(vocab, debug=False)
encoded_test = corpus.encoded_test
# encoded_test = ([10]*2000, [0] * 2000)

output_punc = {0:vocab.decode(0), 1:vocab.decode(1), 2:vocab.decode(2)}

# evaluation
y_pred = model.predict_generator(data_generator(encoded_test, args.batch_size, args.step_size, len(output_punc)),
                                 len(encoded_test[0])//(args.batch_size * args.step_size),
                                 verbose=1)


target_names = ['Blank', 'Comma', 'Period']
y_true = list(np.array(generator_y_true(encoded_test, args.batch_size, args.step_size, len(output_punc))).reshape(-1))

if args.merge_punc:
    print('merge punctuation')
    target_names = ['Blank', 'Punctuation']
    y_true = [x if x == 0 else 1 for x in y_true]
    y_pred = [0 if x[0] > x[1] + x[2] else 1 for x in y_pred.reshape(-1, 3)]
else:
    y_pred = list(y_pred.reshape(-1, 3).argmax(axis=1))

assert len(y_true)== len(y_pred)

print('Confusion Matrix')
print(confusion_matrix(y_true, y_pred))
print("classification report")
print(classification_report(y_true, y_pred, target_names=target_names))

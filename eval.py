from keras.models import load_model
from util import data_generator
from data import CharVocab, Corpus
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
import numpy as np

model = load_model('weight/model_01.hdf5')

# load corpus and vocab
vocab = CharVocab(100000) # 100k
corpus = Corpus(vocab, debug=True)
output_punc = {0:vocab.decode(0), 1:vocab.decode(1), 2:vocab.decode(2)}

# evaluation
y_pred = model.predict_generator(data_generator(corpus.encoded_test, 1, 20, len(output_punc)), 20).reshape(-1, 3).argmax(axis=1)
y_true = corpus.encoded_test[1][:len(y_pred)]

target_names = ['Blank', 'Comma', 'Period']
print('Confusion Matrix')
print(confusion_matrix(y_true, y_pred))
print("classification report")
print(classification_report(y_true, y_pred, target_names=target_names))
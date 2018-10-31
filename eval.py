from keras.models import load_model
from util import data_generator
from data import CharVocab, Corpus
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
import numpy as np

model = load_model('weight/model.hdf5')

# load corpus and vocab
vocab = CharVocab(100000) # 100k
#corpus = Corpus(vocab, debug=True)
output_punc = {0:vocab.decode(0), 1:vocab.decode(1), 2:vocab.decode(2)}

vocab2punc = {'<comma>': '、', '<period>': '。'}
punc_dict = set(['、', '。', '「', '」', '・', '）', '（', '，', '？', '！', '…', '〜', '．', '‐', '『', '』', '―', '：', '“', '”'])

original = '今日はいい天気です今日はいい天気です今日はいい天気です今日はいい天気です今日はいい天気です今日はいい天気です'
input = [x for x in original if x not in punc_dict]
encoded_input = [vocab.encode(x) for x in input]

print(model.predict(np.array(encoded_input).reshape((1,-1))))

pred_y = np.argmax(model.predict(np.array(encoded_input).reshape((1, -1))), axis=2)
print(pred_y)

decoded = []
result = list(pred_y.reshape(-1))
for i in range(len(encoded_input)):
    decoded.append(vocab.decode(encoded_input[i]))
    if vocab.is_punctuation(result[i]):
        decoded.append(vocab2punc[vocab.decode(result[i])])

print(original)
print(''.join(decoded))

'''
# evaluation
y_pred = model.predict_generator(data_generator(corpus.encoded_test, 1, 20, len(output_punc)), 20).reshape(-1, 3).argmax(axis=1)
y_true = corpus.encoded_test[1][:len(y_pred)]

target_names = ['Blank', 'Comma', 'Period']
print('Confusion Matrix')
print(confusion_matrix(y_true, y_pred))
print("classification report")
print(classification_report(y_true, y_pred, target_names=target_names))
'''
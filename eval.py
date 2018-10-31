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

original = '周囲の疑惑の目にさらされながらも、堂々と愛人と同棲し、高級外車を乗りまわしたりと、異様に神経の図太い男だったのだが…'
input = [x for x in original if x not in punc_dict]
encoded_input = [vocab.encode(x) for x in input]

pred = model.predict(np.array(encoded_input).reshape((1,-1)))
#print(pred)
pred_y = np.argmax((pred), axis=2)
#print(pred_y)

decoded = []
result = list(pred_y.reshape(-1))
for i in range(len(encoded_input)):
    decoded.append(vocab.decode(encoded_input[i]))
    if vocab.is_punctuation(result[i]):
        decoded.append(vocab2punc[vocab.decode(result[i])])
    elif pred[0][i][1] > 0.1:
        decoded.append('<c %.2f>' % pred[0][i][1])
    elif pred[0][i][2] > 0.1:
        decoded.append('<p %.2f>' % pred[0][i][2])

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
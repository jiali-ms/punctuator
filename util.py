import os
import csv
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from keras.utils import to_categorical

def punctuation_vocab(corpus_path, punc_vocab_path):
    if os.path.exists(punc_vocab_path):
        print("punctuation vocab already exist, skip ...")
        return

    p_vocab = defaultdict(int)

    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            tokens = line.strip().split(' ')
            for token in tokens:
                if '補助記号' in token:
                    p_vocab[token] += 1

            line = f.readline()

    print(p_vocab)

    with open(punc_vocab_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in p_vocab.items():
            writer.writerow(row)

def data_generator(raw_data, batch_size, num_steps, n_classes):
    X, Y = raw_data
    data_len = len(X)
    batch_len = data_len // batch_size
    data_x = np.zeros([batch_size, batch_len], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_len, n_classes], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = X[batch_len * i:batch_len * (i + 1)]
        data_y[i, :, :] = to_categorical(Y[batch_len * i:batch_len * (i + 1)], num_classes=n_classes)
    epoch_size = batch_len // num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    while True:
        for i in range(epoch_size):
            x = data_x[:, i * num_steps:(i + 1) * num_steps]
            y = data_y[:, i * num_steps:(i + 1) * num_steps]
            yield (x, y)

def generator_y_true(raw_data, batch_size, num_steps, n_classes):
    X, Y = raw_data
    data_len = len(X)
    batch_len = data_len // batch_size
    # data_x = np.zeros([batch_size, batch_len], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        #data_x[i] = X[batch_len * i:batch_len * (i + 1)]
        data_y[i, :] = Y[batch_len * i:batch_len * (i + 1)]

    epoch_size = batch_len // num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    y_true = []

    for i in range(epoch_size):
        # x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        y_true.append(y)

    return y_true

if __name__ == "__main__":
    x = [10] * 1000
    y = list(range(1000))
    a = data_generator((x, y), 32, 10, 3)
    print(np.array(generator_y_true((x, y), 32, 10, 3)).reshape(-1))
import pickle
import os
from tqdm import tqdm

TRAIN_CORPUS_PATH = os.path.join('data', 'train.txt')
DEV_CORPUS_PATH = os.path.join('data', 'dev.txt')
TEST_CORPUS_PATH = os.path.join('data', 'test.txt')

PUNC_VOCAB_PATH = os.path.join('data', 'punc_vocab.csv')
LEXICON_PATH = os.path.join('data', 'lexicon.pkl')

JPN_PERIOD = '補助記号-句点'
JPN_COMMA = '補助記号-読点'
JPN_PUNC = '補助記号'

class Vocab(object):
    def __init__(self, size):
        self.lexicon = pickle.load(open(LEXICON_PATH, 'rb'))[:size]
        # put unk to top.
        # otherwise, if freq if provided, sort with freq
        self.lexicon = [('<unk>', 0)] + self.lexicon
        self.w2i = {x[0]:i for i, x in enumerate(self.lexicon)}
        self.i2w = {v:k for k,v in self.w2i.items()}
        print('vocab with size {} loaded'.format(size))

    def __len__(self):
        return len(self.w2i)

class CharVocab(Vocab):
    def encode(self, c, pos=''):
        if pos and JPN_PUNC in pos:
            if JPN_COMMA in pos:
                return self.c2i['<comma>']
            elif JPN_PERIOD in pos:
                return self.c2i['<period>']
            else:
                assert 'unk punctuation %s' % c

        if c in self.c2i:
            return self.c2i[c]

        return self.c2i['<unk>']

    def decode(self, i):
        if i in self.i2c:
            return self.i2c[i]

        assert 'pass invalid value to vocab %d' % i

    def is_punctuation(self, i):
        if i not in self.i2c:
            return False

        return self.i2c[i] == '<comma>' or self.i2c[i] == '<period>'

    def __init__(self, size):
        super(CharVocab, self).__init__(size)
        self.c2i = {'<blank>': 0, '<comma>': 1, '<period>': 2, '<unk>': 4}

        for item in self.lexicon[len(self.c2i):]:
            word = item[0].split('/')[0]
            pos = item[0].split('/')[-1]
            if JPN_PUNC in pos:
                continue

            for c in word:
                if c not in self.c2i:
                    self.c2i[c] = len(self.c2i)

        self.i2c = {v: k for k, v in self.c2i.items()}
        print('{} chars contained'.format(len(self.c2i)))
        #print(sorted([x for x in self.c2i.keys()]))

    def __len__(self):
        return len(self.c2i)

class Corpus(object):
    def __init__(self, vocab, debug=False):
        self.vocab = vocab
        self.encoded_train = self._encode_corpus(TRAIN_CORPUS_PATH, debug)
        self.encoded_dev = self._encode_corpus(DEV_CORPUS_PATH, debug)
        self.encoded_test = self._encode_corpus(TEST_CORPUS_PATH, debug)

    def _encode_corpus(self, path, debug=False):
        if os.path.exists(path + '.pkl'):
            print('load encoded corpus from dump: %s' % path + '.pkl')
            data = pickle.load(open(path + '.pkl', 'rb'))
            if debug:
                return (data[0][:1024*100], data[1][:2014*100])
            else:
                return data

        encoded_x = []
        encoded_y = []
        print('encode corpus: {}'.format(path))
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if debug:
                lines = lines[:1024*100]
            for line in tqdm(lines):
                tokens = line.strip().split(' ')
                if isinstance(self.vocab, CharVocab):
                    for token in tokens:
                        word = token.split('/')[0]
                        pos = token.split('/')[-1]

                        if JPN_PERIOD in pos or JPN_COMMA in pos:
                            # note that the previous one may already be punctuation
                            # we don't allow continuous punctuation
                            encoded_y[-1] = self.vocab.encode(word, pos)
                        elif JPN_PUNC in pos:
                            # skip unk punctuation
                            continue
                        else:
                            encoded_x += [self.vocab.encode(x) for x in word]
                            encoded_y += [self.vocab.encode('<blank>')] * len(word)

        assert len(encoded_y) == len(encoded_x)

        pickle.dump((encoded_x, encoded_y), open(path + '.pkl', 'wb'))

        return encoded_x, encoded_y

if __name__ == "__main__":
    vocab = CharVocab(100000) # 100k
    corpus = Corpus(vocab, debug=False)
    decoded = []
    train_x, train_y = corpus.encoded_train
    for i in range(100):
        decoded.append(vocab.decode(train_x[i]))
        if vocab.is_punctuation(train_y[i]):
            decoded.append(vocab.decode(train_y[i]))
    print(''.join(decoded))
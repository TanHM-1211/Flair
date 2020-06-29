import codecs
import re
import os
import time


with codecs.open('VN_char.txt', 'r', encoding='utf-8', errors='ignore') as f:
    vn_char = f.read()
with codecs.open('punct.txt', 'r', encoding='utf-8', errors='ignore') as f:
    punct = f.read()

punct = set(punct)


def is_lack_of_info(text):
    cnt_num = 0
    cnt_space = 0
    for c in text:
        if c.isdigit() or c in punct:
            cnt_num += 1
        elif c == ' ':
            cnt_space += 1
    if cnt_num >= 0.3 * (len(text) - cnt_space):
        return True
    return False


def preprocess(sents, dir=None, verbose=200000):
    sep_tokens = r"[\w']+|['!\"#$%&\()*+,-./:;<=>?@\\^_`{}~\[\]]"
    res = []
    if dir is not None:
        print('process', dir)
    i = 0
    for i, sent in enumerate(sents):
        if is_lack_of_info(sent):
            continue
        temp = ' '.join(re.findall(sep_tokens, sent))
        res.append(temp)
        if (i + 1) % verbose == 0:
            print("processing:", i + 1)
    print("processing:", i + 1)
    print('\n')
    res = '\n'.join(res)
    if dir is not None:
        with codecs.open(dir, 'w+', encoding='utf-8',
                         errors='ignore') as f:
            f.write(res)
    return res


def process(dir, limit_per_file=2**30):
    if not os.path.isdir('corpus'):
        os.mkdir("corpus")
    if not os.path.isdir('corpus/train'):
        os.mkdir("corpus/train")

    with codecs.open(dir, 'r', encoding='utf-8', errors='ignore') as f:
        counter = 0
        while True:
            text = f.readline(limit_per_file)
            if len(text) == 0:
                break

            sents = text.split('\n')

            preprocess(sents, "corpus/train/train_split_" + str(counter))
            counter += 1

    with codecs.open("corpus/train/train_split_0", 'r', encoding='utf-8',
                     errors='ignore') as f:
        text = f.read()
        sents = text
        test_size = int(0.7 * len(sents))

    with codecs.open("corpus/test.txt", 'w+', encoding='utf-8',
                     errors='ignore') as f:
        f.write(sents[:test_size])

    with codecs.open("corpus/valid.txt", 'w+', encoding='utf-8',
                     errors='ignore') as f:
        f.write(sents[test_size:])





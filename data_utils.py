# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
import pickle

import numpy as np
from gensim.models import KeyedVectors
from torch.utils.data import Dataset


def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        try:
            tokens = line.rstrip().split()
            if word2idx is None or tokens[0] in word2idx.keys():
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
        except:
            print("Exception while load vectors")
            print(tokens)
            continue
    return word_vec

def load_word2vec(path, word2idx = None):
    word_vec = {}
    w2v_model = KeyedVectors.load_word2vec_format(path, binary=True)
    in_vocb = 0
    words_in_voc = set()
    for word in w2v_model.vocab:
        if word2idx is None or word in word2idx.keys():
            word_vec[word] = w2v_model[word]
            in_vocb += 1
            words_in_voc.add(word)
    # print(set(word2idx.keys()).difference(words_in_voc))
    print("In vocabulary = %s, all words = %s", in_vocb, len(word2idx.keys()))
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.dat'.format(str(embed_dim), type)
    print('loading word vectors...')
    embedding_matrix = np.random.uniform(-0.01, 0.01, [len(word2idx) + 2, embed_dim])
    fname = 'D:/PyCharm Project/vectors/Health_2.5mreviews.s200.w10.n5.v15.cbow.bin' # Path to vectors
    word_vec = load_word2vec(fname, word2idx=word2idx)

    print('building embedding_matrix:', embedding_matrix_file_name)
    for word, i in word2idx.items():
        try:
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        except:
            print("Exception while load vectors")
            continue
    pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


class Tokenizer(object):
    def __init__(self, lower=False, max_seq_len=None, max_aspect_len=None):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.max_aspect_len = max_aspect_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    @staticmethod
    def pad_sequence(sequence, maxlen, dtype='int64', padding='pre', truncating='pre', value=0.):
        x = (np.ones(maxlen) * value).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

    def text_to_sequence(self, text, reverse=False):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        pad_and_trunc = 'post'  # use post padding together with torch.nn.utils.rnn.pack_padded_sequence
        if reverse:
            sequence = sequence[::-1]
        return Tokenizer.pad_sequence(sequence, self.max_seq_len, dtype='int64', padding=pad_and_trunc, truncating=pad_and_trunc)


class ABSADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSADatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                # text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
                # aspect = lines[i + 1].strip()
                # print(i)
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "
        return text

    @staticmethod
    def __read_data__(fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()

            text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            text_left_indices = tokenizer.text_to_sequence(text_left)
            text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right, reverse=True)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            # polarity = int(polarity)+1
            polarity = int(polarity)
            # text = lines[i].lower().strip().replace("$T$", aspect)
            text = lines[i].strip().replace("$T$", aspect)

            data = {
                'text_raw_indices': text_raw_indices,
                'text_raw_without_aspect_indices': text_raw_without_aspect_indices,
                'text_left_indices': text_left_indices,
                'text_left_with_aspect_indices': text_left_with_aspect_indices,
                'text_right_indices': text_right_indices,
                'text_right_with_aspect_indices': text_right_with_aspect_indices,
                'aspect_indices': aspect_indices,
                'polarity': polarity,
                'text': text,
                'aspect': aspect
            }

            all_data.append(data)
        return all_data

    def __init__(self, dataset='twitter', embed_dim=100, max_seq_len=40, fold_num = 1):
        print("preparing {0} dataset...".format(dataset))
        self.fname = {
            'twitter': {
                'train': './datasets/acl-14-short-data/train.raw',
                'test': './datasets/acl-14-short-data/test.raw',
            },
            'restaurant': {
                'train': './datasets/semeval14/Restaurants_Train.xml.seg',
                'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
            },
            'laptop': {
                'train': './datasets/semeval14/Laptops_Train.xml.seg',
                'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
            },
            'cadec': {
                'train': './datasets/cadec/folds/' + str(fold_num) + '/train.txt',
                'test': './datasets/cadec/folds/' + str(fold_num) + '/test.txt',
            },
            'psytar': {
                'train': './datasets/psytar/folds/' + str(fold_num) + '/train.txt',
                'test': './datasets/psytar/folds/' + str(fold_num) + '/test.txt'
            },
            'psytar-cadec':{
                'train': './datasets_correct/psytar-cadec/train.txt',
                'test': './datasets_correct/psytar-cadec/test.txt'
            },
            'psytar-aska_full':{
                'train': './datasets_correct/psytar-aska_full/train.txt',
                'test': './datasets_correct/psytar-aska_full/test.txt'
            },
            'psytar_aska-cadec': {
                'train': './datasets_correct/psytar_aska_full-cadec/train.txt',
                'test': './datasets_correct/psytar_aska_full-cadec/test.txt'
            },
            'cadec-psytar': {
                'train': './datasets_correct/cadec-psytar/train.txt',
                'test': './datasets_correct/cadec-psytar/test.txt'
            },
            'cadec-aska': {
                'train': './datasets_correct/cadec-aska_full/train.txt',
                'test': './datasets/cadec-aska_full/test.txt'
            },
            'psytar_cadec_drugs-cadec': {
                'train': './datasets_correct/psytar_aska_cadec_drugs-cadec/train.txt',
                'test': './datasets_correct/psytar_aska_cadec_drugs-cadec/test.txt'
            },
            'cadec_psytar_drugs-psytar': {
                'train': './datasets_correct/cadec_aska_psytar_drugs-psytar/train.txt',
                'test': './datasets_correct/cadec_aska_psytar_drugs-psytar/test.txt'
            },
            'psytar_webmd(cadec)-cadec': {
                'train': './datasets_correct/psytar_webmd(cadec)-cadec/train.txt',
                'test': './datasets_correct/psytar_webmd(cadec)-cadec/test.txt'
            },
            'cadec_webmd(psytar)-psytar': {
                'train': './datasets_correct/cadec_webmd(psytar)-psytar/train.txt',
                'test': './datasets_correct/cadec_webmd(psytar)-psytar/test.txt'
            },
            'psytar_aska(drugbank)_cadec': {
                'train': './datasets/psytar_drugbank-cadec/train.txt',
                'test': './datasets/psytar_drugbank-cadec/test.txt'
            }

        }
        text = ABSADatesetReader.__read_text__([self.fname[dataset]['train'],self.fname[dataset]['test']])
        tokenizer = Tokenizer(max_seq_len=max_seq_len)
        tokenizer.fit_on_text(text.lower())
        # tokenizer.fit_on_text(text)
        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
        self.train_data = ABSADataset(ABSADatesetReader.__read_data__(self.fname[dataset]['train'], tokenizer))
        print(self.train_data.__getitem__(1))
        self.test_data = ABSADataset(ABSADatesetReader.__read_data__(self.fname[dataset]['test'], tokenizer))





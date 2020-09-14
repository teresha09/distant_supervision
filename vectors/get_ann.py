from nltk.tokenize import wordpunct_tokenize, sent_tokenize
import pymorphy2
from gensim.utils import tokenize
from gensim import utils
from gensim.models import FastText, Word2Vec
from gensim.test.utils import get_tmpfile

morph = pymorphy2.MorphAnalyzer()

def pubmed():
    f = open('data/ru_en_abstracts.txt', encoding='utf-8')
    res = []
    counter = 0
    for line in f:
        if 'null' in line:
            line = line.replace('null', 'None')
        d = eval(line)
        if d['MedlineCitation']['PMID'] == '31627504':
            print('debug')
        if type(d['MedlineCitation']['OtherAbstract']['AbstractText']) != str:
            print(d['MedlineCitation']['PMID'])
        res.append(d['MedlineCitation']['OtherAbstract']['AbstractText'])
    print(counter)
    return res


def microbiology(filename):
    f = open(filename)
    res = []
    names = []
    for line in f:
        d = eval(line)
        if d['name'] in names:
            continue
        names.append(d['name'])
        if d['annotation'] != '':
            if type(d['annotation']) != str:
                print(d['name'])
            res.append(d['annotation'])
    return res


def normalize(texts):
    res_texts = []
    for text in texts:
        if type(text) != str:
            print(text)
            print('dev')
            continue
        tokens = wordpunct_tokenize(text)
        s = []
        for token in tokens:
            p = morph.parse(token)[0]
            s.append(p.normal_form)
        res_texts.append(s)
    return res_texts


def stat(m,p,c):
    all_tokens = []
    counter = 0
    for text in m:
        if type(text) != str:
            continue
        tokens = wordpunct_tokenize(text)
        for token in tokens:
            all_tokens.append(token)
    for text in p:
        if type(text) != str:
            continue
        tokens = wordpunct_tokenize(text)
        for token in tokens:
            all_tokens.append(token)
    for text in c:
        if type(text) != str:
            continue
        tokens = wordpunct_tokenize(text)
        for token in tokens:
            all_tokens.append(token)
    print(len(all_tokens))
    print(len(set(all_tokens)))

class MyIter(object):
    def __init__(self, pubmed):
        self.pubmed = pubmed
        # self.micbio = micbio
        # self.cl = cl
        # self.articles2 = articles2


    def __iter__(self):
        for sent in self.pubmed:
            yield sent
        # for sent in self.micbio:
        #     yield sent
        # for sent in self.cl:
        #     yield sent


mb = normalize(microbiology("data/microbiology.json"))
#norm_pm = normalize(pubmed())
#cyberl = normalize(microbiology("data/articles.txt"))
#a2 = normalize(microbiology("data/articles2.txt"))


#m = microbiology()
#p = pubmed()
#cl = microbiology(True)

# print(len(m))
# print(len(p))
# print(len(cl))
# stat(m,p,cl)

path = get_tmpfile("word2vec.model")
model4 = Word2Vec(sentences=MyIter(mb), size=100, window=5, min_count=1, workers=4)
#model4.build_vocab(sentences=MyIter(mb, norm_pm, cyberl, a2))
total_examples = model4.corpus_count
model4.train(sentences=MyIter(mb), total_examples=total_examples, epochs=1)
model4.save(path)

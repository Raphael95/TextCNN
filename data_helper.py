import numpy as np
from nltk.tokenize import WordPunctTokenizer
import pandas as pd
from gensim.models import word2vec
import re
import os
import time
import jieba
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class preprocess(object):

    # load te data
    def load_data(self):
        print("loading the data... ...")
        excel = pd.read_excel('./data/train_data.xls', 'news_data_de')
        column1 = ['context']
        column2 = ['theme']
        feature = pd.DataFrame(excel, columns=column1)
        label = pd.DataFrame(excel, columns=column2)
        x = np.array(feature)
        y = np.array(label)
        x = x[0: 6000, 0]
        y = y[0: 6000, 0]
        print("===============================")
        return x, y

    # tokenize the word from all document and write to a file
    def tokenize_word(self, document):
        print("tokenizing the word on the whole document, it will takes a long time ,please wait... ...")
        if os.path.exists(r'./word.txt'):
            pass
        else:
            for sentence in document:
                sentence = sentence.replace("\n", " ")
                sentence = delete_number(sentence)
                segement = jieba.cut(sentence.strip(), cut_all=True)
                stopwords = stopword_lists('./chinese_stopwords.txt')
                outstr = ''
                for word in segement:
                    if word not in stopwords:
                        outstr += word
                        outstr += " "
                outstr += "\n"
                with open('./word.txt', 'a+') as f:
                    f.writelines(outstr)
                    f.close()
            print("write the word done !")
            print("============================")


    # using gensim to generate word2vec model
    def generate_word2vec(self):
        print("generate the word2vec model... ...")

        # to check is the model has existed
        if os.path.exists(r'./word_vec.model'):
            model = word2vec.Word2Vec.load(r'./word_vec.model')
        else:
            word_document = word2vec.Text8Corpus(r"./word.txt")
            model = word2vec.Word2Vec(word_document, min_count=2, size=100)
            model.save(r'./word_vec.model')
            print("well done !")
            print("============================")
        return model

    # modify the word2vec, inserting zero element to the first line
    # return new vocabulary list, word2vec
    def modify_word2vec(self, model):
        vocab = list(model.wv.vocab)
        vocab.insert(0, 'NONE')
        word_vec = np.array(model.wv.syn0)
        #zero_line = [np.random.uniform(-0.25, 0.25, 100)]   # 均匀分布初始化 index 是 0 的词语的词向量
        zero_line = [[0 for i in range(100)]]
        word_vec = np.concatenate((zero_line, word_vec), axis=0)
        return vocab, word_vec


    # calculate the sentence_word
    def cal_sentence_word(self, vocab):
        print("begin to calculate the word vector")
        vocab_lists = vocab
        token_document = [sentence.strip() for sentence in open('./word.txt', encoding='utf-8').readlines()]
        input_data = []
        for i in token_document:
            sentence_index = []
            split_word_sentence = WordPunctTokenizer().tokenize(i)
            for j in split_word_sentence:
                try:
                    sentence_index.append(vocab_lists.index(j))
                except ValueError:
                    continue

            input_data.append(sentence_index)
        print("complete !")
        print("============================")
        return input_data

    # calculate the max length of sentence in all document
    def max_len_sentence(self, input_data):
        max_len = 0
        for sentence in input_data:
            sentence_len = len(sentence)
            if sentence_len > max_len:
                max_len = sentence_len
        return max_len


    # making length of every sentence same, if length of sentence is not up to max_len , makeing up zero
    def makeup_sentence(self, input_data, max_len):
        full_sentence = []
        for sentence in input_data:
            if len(sentence) < max_len:
                i = max_len - len(sentence)
                zero_element = [0 for j in range(i)]
                sentence = np.concatenate((sentence, zero_element), axis=0)
            else:
                sentence = sentence[0: max_len]
            full_sentence.append(sentence)
        full_sentence = np.array(full_sentence)
        return full_sentence

    # one_hot encoder the label
    def one_hot_encoder(self, y):
        label = LabelEncoder().fit_transform(y)
        label = OneHotEncoder(sparse=False).fit_transform(label.reshape((-1, 1)))
        return label


# to delete the number from sentence
def delete_number(word):
    return re.sub("\d+", "", word)


# define the stopwords
def stopword_lists(file_path):
    stopwords = [line.strip() for line in open(file_path, 'r', encoding='utf-8').readlines()]
    return stopwords

# to generate a enumerate for batch_x, batch_y
def generate_batch(x, y, loop_size, batch_size):
    index = []
    all_sentence_len = np.shape(x)[0]
    while loop_size:
        index_begin = np.random.randint(0, all_sentence_len - batch_size + 1)
        if index_begin not in index:
            index_end = index_begin + batch_size
            batch_x = x[index_begin: index_end]
            batch_y = y[index_begin: index_end]
            yield batch_x, batch_y
            index.append(index_begin)
            loop_size -= 1





if __name__ == '__main__':

    start_time = time.time()
    preprocess = preprocess()
    x, y = preprocess.load_data()
    preprocess.tokenize_word(x)
    model = preprocess.generate_word2vec()
    vocab, word_vec = preprocess.modify_word2vec(model)
    input_data = preprocess.cal_sentence_word(vocab)
    max_len = preprocess.max_len_sentence(input_data)
    input_data = preprocess.makeup_sentence(input_data, 2000)
    print("input_data :", input_data)
    print("input_data shape is :", input_data.shape)
    print("vocabulary num : ", np.shape(word_vec))
    label = preprocess.one_hot_encoder(y)
    print("label is :", label)
    print("label shape is :", np.shape(label))
    end_time = time.time()

    batch = generate_batch(input_data, label, 500, 10)
    for i in range(500):
        batch_x, batch_y = batch.__next__()
        print("batch_x shape is :", np.shape(batch_x))
        print("batch_x : ", batch_x)
        print("batch_y shape is :", np.shape(batch_y))
        print("batch_y : ", batch_y)
        print("====================================")

    print("运行时间 : ", (end_time - start_time) / 60)
#!/usr/bin/env python
# coding: utf-8

import MeCab
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import normalize


tagger = MeCab.Tagger("-Ochasen")

def tokenaize(doc):

    node = tagger.parseToNode(doc)

    result = []
    while node:

        meta = node.feature.split(",")[0]
        if meta == "名詞" or meta == "動詞" or meta == "形容詞":
            result.append(node.feature.split(",")[6])

        node = node.next

    return result


def count_vectorizer(corpus):

    documents = []
    for words in corpus:

        words = tokenaize(words)
        documents.append(words)

    print(documents)

    vocab = defaultdict()
    vocab.default_factory = vocab.__len__

    for doc in documents:

        feature_counter = {}
        for feature in doc:

            feature_index = vocab[feature]
            if feature_index not in feature_counter:
                feature_counter[feature_index] = 1
            else:
                feature_counter[feature_index] += 1

    sorted_feature = sorted(vocab.items())
    for new_value, term in enumerate(sorted_feature):
        vocab[term] = new_value

    X = np.zeros(shape=(len(corpus), len(sorted_feature)), dtype=int)
    for index, document in enumerate(documents):

        for word in document:

            if word in vocab.keys():
                X[index, vocab[word]] += 1

    return X


def tf_idf(count_vectors):

    smooth_idf = True
    norm_idf = True

    wcX = np.array(count_vectors)
    N = wcX.shape[0]

    tf = np.array([wcX[i, :] / np.sum(wcX, axis=1)[i] for i in range(N)])

    df = np.count_nonzero(wcX, axis=0)
    idf = np.log((1 + N) / (1 + df)) + 1 if smooth_idf else np.log(N / df)

    tfidf = normalize(tf * idf) if norm_idf else tf * idf

    return tfidf


if __name__ == "__main__":

    corpus = ["僕の趣味は釣りです。この前、宍道湖で70㎝のシーバスを釣りあげました。",
              "僕はボカロPもどきです。たまにコミケにアルバムを出展したりします。"]

    count_vector = count_vectorizer(corpus)
    tfidf = tf_idf(count_vector)

    print(tfidf)

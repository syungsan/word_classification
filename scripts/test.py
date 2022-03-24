#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.simplefilter("ignore")

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def preprocess(data, tokenizer, maxlen=280):
    return(pad_sequences(tokenizer.texts_to_sequences(data), maxlen=maxlen))


def predict(sentences, graph, emolabels, tokenizer, model, maxlen):
    preds = []
    targets = preprocess(sentences, tokenizer, maxlen=maxlen)
    with graph.as_default():
        for i, ds in enumerate(model.predict(targets)):
            preds.append({
                "sentence":sentences[i],
                "emotions":dict(zip(emolabels, [str(round(100.0*d)) for d in ds]))
            })
    return preds


def load(path):
    model = load_model(path)
    graph = tf.get_default_graph()
    return model, graph


if __name__ == "__main__":

    maxlen = 280
    model, graph = load("../models/model_2018-08-28-15-00.h5")

    with open("../models/tokenizer_cnn_ja.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    emolabels = ["happy", "sad", "disgust", "angry", "fear", "surprise"]

    print("Model Load Complete\n")
    print("Write Sentence!\n")

    while True:

        sentence = [input()]

        results = predict(sentence, graph, emolabels, tokenizer, model, maxlen)
        preduct_emotions = []

        for text_with_emotion in results:
            emotions = text_with_emotion['emotions']

            probabilitys = []
            for emotion_item in emotions.items():
                probabilitys.append(float(emotion_item[1]))

            probability_distributions = []
            for probability in probabilitys:
                probability_distributions.append(probability / sum(probabilitys))

            # 感情のスコアがもっとも高いものだけを抽出
            max_emos = [max_emotions[0] for max_emotions in emotions.items() if max_emotions[1] == max(
                emotions.items(), key=(lambda emotion: float(emotion[1])))[1]]

            for max_emo in max_emos:
                preduct_emotions.append(max_emo)

            output = "WORD_EMOTION,%s" % preduct_emotions[0]
            for probability_distribution in probability_distributions:
                output += ",%s" % str(probability_distribution)

            print(output + "\n")
            print("Write Sentence!\n")

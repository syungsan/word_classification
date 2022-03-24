#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from os.path import join
from sklearn.utils import shuffle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from keras.layers import Dense, Embedding, Flatten
from keras.layers import SpatialDropout1D
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import Sequential

from sklearn.metrics import classification_report
import numpy as np


emotions = ["happy", "sad", "disgust", "angry", "fear", "surprise"]
dir_path = "../data"

size = 18000 # 60000

df = []
for i, es in enumerate(emotions):

    if isinstance(es, list):

        for e in es:

            data = shuffle(pd.read_json(join(dir_path, "{}.json".format(e)))).iloc[:int(size/len(es))]
            data['label'] = i

            df.append(data)
    else:
        data = shuffle(pd.read_json(join(dir_path, "{}.json".format(es)))).iloc[:int(size)]
        data['label'] = i

        df.append(data)

df = pd.concat(df)
df = shuffle(df)

X = df['text']
y = df['label']

print(df.shape)

max_features = 3000 # 10000
maxlen = 280

def preprocess(data, tokenizer, maxlen=280):
    return(pad_sequences(tokenizer.texts_to_sequences(data), maxlen=maxlen))


y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y)

tokenizer = Tokenizer(num_words=max_features, filters="", char_level=True)
tokenizer.fit_on_texts(list(X_train))

X_train = preprocess(X_train, tokenizer, maxlen)
X_test = preprocess(X_test, tokenizer, maxlen)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)

print(X_train.shape, X_val.shape, X_test.shape)

model = Sequential()
model.add(Embedding(max_features, 150, input_length=maxlen))
model.add(SpatialDropout1D(0.2))
model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(6, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

epochs = 15
batch_size = 1000

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
model.save("../model/model_last.h5")

# show Accuracy and Loss History
import compare_TV
compare_TV.compare_TV(history)

y_preds = model.predict(X_test)
y_preds = np.argmax(y_preds, axis=1)
y_true = np.argmax(y_test, axis=1)

emolabels = []

for e in emotions:

    if isinstance(e, list):
        emolabels.append(e[0])
    else:
        emolabels.append(e)

print(classification_report(y_true, y_preds, target_names=emolabels))

examples = [
    "まじきもい、あいつ",
    "今日は楽しい一日だったよ",
    "ペットが死んだ、実に悲しい",
    "ふざけるな、死ね",
    "ストーカー怖い",
    "すごい！ほんとに！？",
    "葉は植物の構成要素です。",
    "ホームレスと囚人を集めて革命を起こしたい"
    "プレゼントをありがとう！"
]

targets = preprocess(examples, tokenizer, maxlen=maxlen)

print('\t'.join(emolabels))

for i, ds in enumerate(model.predict(targets)):
    print('\t'.join([str(round(100.0*d)) for d in ds]))

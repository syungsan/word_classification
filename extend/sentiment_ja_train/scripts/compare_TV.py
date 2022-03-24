#!/usr/bin/env python
# coding: utf-8

def compare_TV(history):
    import matplotlib.pyplot as plt

    # Setting Parameters
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    # 1) Accracy Plt
    plt.plot(epochs, acc, 'bo', label='training acc')
    plt.plot(epochs, val_acc, 'r', label='validation acc')
    plt.title('Training and Validation acc')
    plt.legend()
    plt.savefig("../data/accuracy.svg", format="svg")

    plt.figure()

    # 2) Loss Plt
    plt.plot(epochs, loss, 'bo', label='training loss')
    plt.plot(epochs, val_loss, 'r', label='validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.savefig("../data/loss.svg", format="svg")

    plt.show()

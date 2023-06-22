import numpy as np


def c10(amount):
    return float(amount)

def c01():
    return 100.0

def single_loss(true_label, predicted_label, amount):
    assert((predicted_label) == 0 or predicted_label==1)
    if abs(true_label - predicted_label)<1e-6:
        return 0.0
    if true_label == 1:
        return c10(amount)
    return c01()

def f_losses(true_label, predictions, weights, amount, sogliazza=False):
    l = []
    for p in predictions:
        l.append(single_loss(true_label, p, amount))
    if not sogliazza:
        l.append(np.sum(weights*l))
    else:
        w_pred = np.sum(predictions*weights)
        l.append(single_loss(true_label, w_pred>0.5, amount))
    return l

def grad_loss(true_label, predictions, amount):
    return (c01()*(1-true_label)-c10(amount)*true_label)*predictions
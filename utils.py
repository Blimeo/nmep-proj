import torch
import numpy as np
import matplotlib.pyplot as plt

def accuracy(model, dataset, device, n_train=None):
    count = 0
    num_examples = 0
    bs = dataset.batch_size
    n_examples_seen = 0
    for batch in dataset:
        if n_train and n_examples_seen >= n_train:
            break
        inputs, labels = batch[0].to(device), batch[1].to(device)
        n_examples_seen += len(inputs)
        out = model(inputs)
        count += torch.sum(torch.argmax(out, dim=1) == labels)
        num_examples += len(labels) * 1.
    return count.item() / num_examples


def plot_save(train_x, train_y, filename, val_x=None, val_y=None):
    plt.plot(train_x, train_y, label='train')
    if val_x:
        plt.plot(val_x, val_y, label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(filename)
    plt.clf()

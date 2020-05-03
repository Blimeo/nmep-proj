import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torchattacks import FGSM, PGD
from adversarial import *

def accuracy(model, dataset, device, n_train=None):
    count = 0
    num_examples = 0
    for batch in dataset:
        if n_train and num_examples >= n_train:
            break
        inputs, labels = batch[0].to(device), batch[1].to(device)
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

def display_im(im):
    # https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3, undoes the normalization
    invertTranspose = transforms.Compose([
                                transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.2023, 1/0.1994, 1/0.2010 ]),
                                transforms.Normalize(mean = [ -0.4914, -0.4822, -0.4465 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    im = invertTranspose(im)
    # print(im, 'help')
    im = im.permute(1,2,0)
    plt.imshow(im)
    plt.show()


def fgsm_accuracy(model, dataset, device):
    count = 0
    num_examples = 0
    attack = FGSM(model)
    for batch in dataset:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        inputs = attack(inputs, labels)
        out = model(inputs)
        count += torch.sum(torch.argmax(out, dim=1) == labels)
        num_examples += len(labels) * 1.
    return count.item() / num_examples

def pgd_accuracy(model, dataset, device, eps=8, steps=7, alpha=0.1):
    count = 0
    num_examples = 0
    # attack = PGD(model, eps=eps, iters=steps, alpha=alpha)
    for batch in dataset:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        # inputs = attack(inputs, labels)
        inputs = pgd(inputs, labels, model, stepsize=alpha, eps=eps, steps=steps)
        out = model(inputs)
        count += torch.sum(torch.argmax(out, dim=1) == labels)
        num_examples += len(labels) * 1.
    return count.item() / num_examples

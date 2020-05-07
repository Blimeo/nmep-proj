import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torchattacks import FGSM, PGD
from adversarial import *

def accuracy(model, dataset, device):
    count = 0
    num_examples = 0
    for batch in dataset:
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
    invertTransform = transforms.Compose([
                                transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    im = invertTransform(im)
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
        model.zero_grad()
        inputs = attack(inputs, labels)
        out = model(inputs)
        count += torch.sum(torch.argmax(out, dim=1) == labels)
        num_examples += len(labels) * 1.
    return count.item() / num_examples

def pgd_accuracy(model, dataset, device, eps=8, steps=7, alpha=0.1):
    count = 0
    num_examples = 0
    for batch in dataset:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        model.zero_grad()
        adv = pgd(inputs, labels, model, stepsize=alpha, eps=eps, steps=steps)
        out = model(adv)
        count += torch.sum(torch.argmax(out, dim=1) == labels)
        num_examples += len(labels) * 1.
    return count.item() / num_examples

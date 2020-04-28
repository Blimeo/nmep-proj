import torch
import numpy as np

def accuracy(model, dataset, device):
    count = 0
    num_examples = 0

    for batch in dataset:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        out = model(inputs) # to implement adversarial training, perterb batch before running model
        count += torch.sum(torch.argmax(out, dim=1) == labels)
        num_examples += len(labels) * 1.
        
    return count.item() / num_examples

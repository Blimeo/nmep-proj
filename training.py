import torch
import torch.nn as nn
import torchvision
import numpy as np
from tqdm import tqdm
from utils import *


seed = 0
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def train(train_dataset, val_dataset, model, epochs, lr):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum = 0.9, weight_decay=1)

    loss =  nn.CrossEntropyLoss() # cross entropy

    for epoch in range(epochs):
        total_loss = 0
        count = 0
        for batch in tqdm(train_dataset):

            model.zero_grad()
            inputs, labels = batch[0].to(device), batch[1].to(device)
            out = model(inputs) # to implement adversarial training, perterb batch before running model
            l = loss(out, labels)
            total_loss += l
            count += 1
            grad = l.backward()
            optimizer.step()

        model.eval()
        print('Epoch ', epoch, 'Training Loss ', total_loss.item() / count, 'Training Accuracy ', accuracy(model, train_dataset, device),
        'Validation Accuracy ', accuracy(model, val_dataset, device))
        model.train()
        # print validation loss here, per epoch


transform = torchvision.transforms.ToTensor()

train_data = torchvision.datasets.CIFAR10('data/', download=True, train=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_data,
                                          batch_size=256,
                                          shuffle=True)

test_data = torchvision.datasets.CIFAR10('data/', download=True, train=False, transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_data,
                                          batch_size=256,
                                          shuffle=True)


model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
model.to(device)
model.train()
# print(model)
train(train_dataloader, test_dataloader, model, 100, 0.001)

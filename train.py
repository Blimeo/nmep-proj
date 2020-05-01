import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import numpy as np
from tqdm import tqdm
from utils import *
from adversarial import *


seed = 0
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)


def train(train_dataset, val_dataset, model, epochs, lr, adversarial=False, n_train=None):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=5e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

    loss =  nn.CrossEntropyLoss() # cross entropy
    plotted_train_epochs = []
    train_accuracies = []
    test_accuracies = []
    train_losses = []

    for epoch in range(epochs):
        total_loss = 0
        count = 0
        total_examples_seen = 0
        for batch in tqdm(train_dataset):
            if n_train and total_examples_seen >= n_train:
                break
            inputs, labels = batch[0].to(device), batch[1].to(device)
            total_examples_seen += len(inputs)
            if adversarial:
                # inputs = pgd(inputs, labels, model, loss, 0.1, 0.5, 7)
                model.zero_grad()
                out = model(inputs)
            else:
                model.zero_grad()
                out = model(inputs) # to implement adversarial training, perterb batch before running model
            l = loss(out, labels)
            total_loss += l
            count += 1
            grad = l.backward()
            optimizer.step()

        model.eval()
        train_loss =  total_loss.item() / count
        train_losses.append(train_loss)
        test_acc = accuracy(model, val_dataset, device)
        test_accuracies.append(test_acc)
        if epoch % 5 == 0: # calculating accuracy over all training examples takes a few seconds, makes training annoying
            train_acc = accuracy(model, train_dataset, device, n_train)
            train_accuracies.append(train_acc)
            plotted_train_epochs.append(epoch)
            print('Epoch ', epoch, 'Training Loss ', train_loss, 'Training Accuracy ', train_acc, 'Validation Accuracy ', test_acc)
        else:
            print('Epoch ', epoch, 'Training Loss ', train_loss, 'Validation Accuracy ', test_acc)

        plot_save(plotted_train_epochs, train_accuracies, 'accuracy.png', range(epoch + 1), test_accuracies)
        plot_save(range(epoch + 1), train_losses, 'loss.png')
        model.train()
        scheduler.step()
        # print validation loss here, per epoch


transform = transforms.Compose([transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.CIFAR10('data/', download=True, train=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_data,
                                          batch_size=256,
                                          shuffle=True)

test_data = torchvision.datasets.CIFAR10('data/', download=True, train=False, transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_data,
                                          batch_size=1024, # just for test accuracy
                                          shuffle=True)


model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False)
model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
model.to(device)
model.train()
# print(model)
train(train_dataloader, test_dataloader, model, 100, 0.001, adversarial=False)
# train(train_dataloader, test_dataloader, model, 100, 0.01)

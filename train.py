import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import numpy as np
from tqdm import tqdm
from utils import *
from adversarial import *
import torchattacks
import os


seed = 0
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def train(train_dataset, val_dataset, model, epochs, lr, model_name, adversarial=False, checkpoint=True):

    print('Training {0} for {1} epochs'.format(model_name, epochs))
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=5e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350], gamma=0.1)

    loss =  nn.CrossEntropyLoss() # cross entropy
    plotted_train_epochs = []
    train_accuracies = []
    test_accuracies = []
    train_losses = []
    # pgd_attack = torchattacks.PGD(model, eps=0.5, alpha=0.1, iters=5)

    for epoch in range(epochs):
        total_loss = 0
        count = 0
        total_examples_seen = 0
        for batch in tqdm(train_dataset):
            inputs, labels = batch[0].to(device), batch[1].to(device)
            total_examples_seen += len(inputs)
            if adversarial:
                # inputs = pgd_attack(inputs, labels).to(device)
                inputs = pgd(inputs, labels, model, stepsize=10*2.5/7, eps=10, steps=7, constraint='l_2')
                # inputs = fgsm(inputs, labels, model, eps=.08)
                optimizer.zero_grad()
                out = model(inputs)
            else:
                optimizer.zero_grad()
                out = model(inputs) # to implement adversarial training, perterb batch before running model
            l = loss(out, labels)
            total_loss += l
            count += 1
            grad = l.backward()
            optimizer.step()

        # tracking
        model.eval()
        with torch.no_grad():
            train_loss =  total_loss.item() / count
            train_losses.append(train_loss)
            test_acc = accuracy(model, val_dataset, device)
            test_accuracies.append(test_acc)
            if epoch % 5 == 0: # calculating accuracy over all training examples takes a few seconds, makes training annoying
                train_acc = accuracy(model, train_dataset, device)
                train_accuracies.append(train_acc)
                plotted_train_epochs.append(epoch)
                print('Epoch ', epoch, 'Training Loss ', train_loss, 'Training Accuracy ', train_acc, 'Validation Accuracy ', test_acc)
            else:
                print('Epoch ', epoch, 'Training Loss ', train_loss, 'Validation Accuracy ', test_acc)

            root = 'training_results/' + model_name + '/'
            if not os.path.exists(root):
                os.mkdir(root)
            plot_save(plotted_train_epochs, train_accuracies, root + 'accuracy.png', range(epoch + 1), test_accuracies)
            plot_save(range(epoch + 1), train_losses, root + 'loss.png')
        model.train()
        scheduler.step()

        if checkpoint and epoch % 5 == 0:
            root = 'models/' + model_name + '/'
            if not os.path.exists(root):
                os.mkdir(root)
            torch.save(model.state_dict(), root + str(epoch))

    # done training
    torch.save(model.state_dict(), root + 'final')



# transform = transforms.Compose([transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
#                                 transforms.RandomHorizontalFlip(p=0.5),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
# train_transform = transforms.Compose([transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
#                                 transforms.RandomHorizontalFlip(p=0.5),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
# ])
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                                     # , transforms.Lambda( lambda x: x / 255.)
])

test_transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                                     # , transforms.Lambda( lambda x: x / 255.)
])

train_data = torchvision.datasets.CIFAR10('data/', download=True, train=True, transform=train_transform)
train_dataloader = torch.utils.data.DataLoader(train_data,
                                          batch_size=256,
                                          shuffle=True)

test_data = torchvision.datasets.CIFAR10('data/', download=True, train=False, transform=test_transform)
test_dataloader = torch.utils.data.DataLoader(test_data,
                                          batch_size=1024, # just for test accuracy
                                          shuffle=False)


from models import *
model = ResNet('18')
model.to(device)
model.train()
# print(model)
# train(train_dataloader, test_dataloader, model, 100, 0.001, adversarial=False)
train(train_dataloader, test_dataloader, model, 350, 0.1, adversarial=True, model_name='resnet18_l2eps=10')
# train(train_dataloader, test_dataloader, model, 25, 0.1, adversarial=False, model_name='resnet18_normal')

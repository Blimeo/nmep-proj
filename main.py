import torch
import torchvision.transforms as transforms
from utils import *
from models import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


test_transform = transforms.Compose([transforms.ToTensor(),
               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                        # , transforms.Lambda( lambda x: x / 255.)
])
test_data = torchvision.datasets.CIFAR10('data/', download=True, train=False, transform=test_transform)
test_dataloader = torch.utils.data.DataLoader(test_data,
                                          batch_size=1024, # just for test accuracy
                                          shuffle=True)

cifar_labels = {0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}


model = ResNet('18').to(device)
# model.load_state_dict(torch.load('models/resnet18_l2stepsize=0.1eps=0.5steps=7/8'))
model.load_state_dict(torch.load('models/resnet18_l2eps=10steps=7/10'))

model.eval()
#
# it = iter(test_dataloader)
# image, label = it.next()
# image, label = image[0].to(device), label[0].view(1).to(device)
# display_im(image.cpu())

# pgd(inputs, labels, model, stepsize=0.1, eps=0.5, steps=5)
# adv = pgd(image.view(1,3,32,32), label, model, stepsize=0.1, eps=40, steps=5)
#
# print(torch.max(image - adv))
# print(adv.shape)
#
# print(torch.argmax(model(adv)))
# display_im(adv.view(3,32,32).cpu())


test_acc = accuracy(model, test_dataloader, device)
fgsm_acc = fgsm_accuracy(model, test_dataloader, device)
pgd_acc = pgd_accuracy(model, test_dataloader, device, eps=10, steps=7, alpha=2.5 * 10 / 7)
print('Normal Test Accuray:', test_acc)
print('FGSM Test Accuracy:', fgsm_acc)
print('PGD Test Accuracy:', pgd_acc)

import torch
import torch.nn as nn
import torchvision

class CNN(nn.Module): # random basic model for testing things
    def __init__(self):
        super(CNN, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(3,16,5),
            nn.ReLU(),
            nn.Conv2d(16,32,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(64*4*4,100),
            nn.ReLU(),
            nn.Linear(100,10)
        )

    def forward(self,x):
        print(x.size())
        out = self.layer(x)
        out = out.view(-1,64*4*4)
        out = self.fc_layer(out)

        return out

class ResNet(nn.Module):
    def __init__(self, version):
        super(ResNet, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet' + version, pretrained=False)
        self.resnet.fc = nn.Linear(in_features=self.resnet.fc.in_features, out_features=10, bias=True)


    def forward(self,x):
        out = self.resnet(x)
        return out

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.mobile = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2' , pretrained=False)
        self.mobile.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=False), nn.Linear(1280, 10, bias=True))
    def forward(self,x):
        out = self.mobile(x)
        return out
# resnet.conv1 = nn.Conv2d(3, channels1, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# resnet.bn1 = nn.BatchNorm2d(channels1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# resnet.layer1.conv1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
# resnet.layer1.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# resnet.layer1.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
# resnet.layer1.bn2 = bb.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

from __future__ import print_function
import torch
from torch import nn, optim
import torch.nn.functional as F

import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torchinfo import summary
import sys

# Implementation of residual block to be reused.
class FCN(nn.Module):
    
    def __init__(self, num_classes,in_channels=1, task=None,train_type=None,dropout=0.5):
        
        super(FCN, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_type= train_type
        self.dropout_p = dropout
        self.task = task
        
        backbone = models.resnet50(pretrained=False)

        self.conv1 = nn.Conv2d(1,64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.dropout = nn.Dropout2d(p=self.dropout_p)
        

        self.update_task(self.task,self.num_classes)
        self.initialize_weights()



    # Function for randomly initializing weights.
    def initialize_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def update_task(self,new_task, new_classes):

        self.task = new_task
        self.new_classes = new_classes

        if self.task == None or self.task == "rotation" or self.task == "jigsaw":
            self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.classifier = nn.Linear(2048, self.new_classes)

        elif self.task == "segmentation" or self.task=='presentation':

            self.segmenter = nn.Sequential(
                nn.Conv2d(256+2048,256,kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256,self.new_classes,kernel_size=3, stride=1, padding=1, bias=False))

        else:
            raise ValueError(f'Not recognized task {self.task}')   
        


    def forward(self, x):

        assert self.task== None or self.task=='segmentation' or self.task=='rotation' or self.task=='hog' or self.task=='jigsaw' or self.task=='inpainting'or self.task=='presentation', "Not recognized task"
        
        # First convolution - network entrance - here keeping size
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.maxpool(out)

        activ1 = self.layer1(out)
        activ1 = self.dropout(activ1)
        activ2 = self.layer2(activ1)
        activ2 = self.dropout(activ2)
        activ3 = self.layer3(activ2)
        activ3 = self.dropout(activ3)
        activ4 = self.layer4(activ3)
        activ4 = self.dropout(activ4)

        if self.task == None or self.task == "rotation" or self.task == "jigsaw":
            # Adaptive Average Pooling for consistent output size of 1x1.
            out = self.avgpool(activ4)
            # Linearizing trailling dimensions.
            out = out.view(x.size(0), -1)
            # Inference layer.
            out = self.classifier(out)


        elif self.task == "segmentation" or self.task=='presentation':
            inter1 = F.interpolate(activ1, size=x.size()[2:] , mode='bilinear') #interpolation and concat wont appear
            inter4 = F.interpolate(activ4, size=x.size()[2:] , mode='bilinear') #on architecture for they are not default, nor learnable
            out = torch.cat([inter1,inter4],dim=1)
            out = self.segmenter(out)


        else:
            raise ValueError(f'Not recognized task {self.task}')      

        
        return out
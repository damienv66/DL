import torch
import torch.nn as  nn
import torch.nn.functional as F
import torch
import  torchvision
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      
      x = self.batch_norm2(self.conv2(x))
      
      if self.i_downsample is not None:
          
          identity = self.i_downsample(identity)
      #print(x.shape)
      #print(identity.shape)
      x += identity
      x = self.relu(x)
      return x


        
        
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, K, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = K
        
        self.conv1 = nn.Conv2d(num_channels, K, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(K)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=K)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=K*2, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=K*4, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=K*8, stride=2)
        
        #self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        #self.fc = nn.Linear(K*8*ResBlock.expansion, num_classes)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        #x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            #print(planes*ResBlock.expansion, self.in_channels)
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

        
        
def ResNet50(K,channels=3):
    return ResNet(Block, [1,1,1,1],K, channels)
    
def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)

class ResNet18_perso(nn.Module):
    def __init__(self, pool='average', pretrain=True):
        super(CNN, self).__init__()
        weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrain else None
        self.ResNet=torchvision.models.resnet18(pretrained=True)
        self.layers = list(self.ResNet.children())[:2]
        self.Partial_ResNet = nn.Sequential(*self.layers)
        self.flatten= nn.Flatten()
        self.pool=pool 
        
        
    def forward(self, bag_images):
       x = self.Partial_ResNet(bag_images)
       x = self.flatten(x)
       if self.pool=='average':
           x = torch.mean(x, dim=0)
       else:
           x = torch.max(x, dim=0)
       return x  
    
class CNN(nn.Module):
    def __init__(self, K, num_classes=2, channels=3):
        super(CNN, self).__init__()
        self.ResNet=ResNet50(K, channels)
        self.fc = nn.Linear(K*8*7*7, num_classes)    
        
        
    def forward(self, bag_images):
       x = self.ResNet(bag_images)
       print(x.shape, 'sortie res net')
       x = torch.mean(x, dim=0)
       print(x.shape, 'sortie  mean')
       #x = F.relu(x)
       
       x = self.fc(x)
       print(x.shape, 'sortie  linear')
       #print(x.shape)
       #x = F.relu(x)
       x = torch.log_softmax(x,dim=0)
       print(x.shape, 'sortie  softmax')
       #print(x)
       return x 
  

class CNN(nn.Module):
    def __init__(self, K, num_classes=2, channels=3):
        super(CNN, self).__init__()
        self.ResNet=ResNet50(K, channels)
        self.fc = nn.Linear(K*8*7*7, num_classes)    
        
        
    def forward(self, bag_images):
       x = self.ResNet(bag_images)
       print(x.shape, 'sortie res net')
       x = torch.mean(x, dim=0)
       print(x.shape, 'sortie  mean')
       #x = F.relu(x)
       
       x = self.fc(x)
       print(x.shape, 'sortie  linear')
       #print(x.shape)
       #x = F.relu(x)
       x = torch.log_softmax(x,dim=0)
       print(x.shape, 'sortie  softmax')
       #print(x)
       return x 




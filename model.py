import torch 
import torch.nn as nn #
import torch.nn.functional as F # various activation functions for model

# 일부러 좀 낮은 classifier 를 쓸 필요가 있음.
# Ra GAN Loss 도 사용해야할수도?
class netD(nn.Module):
    def __init__(self):
        super(netD, self).__init__() 
        # an affine operation: y = Wx + b
        self.fc0 = nn.Linear(28*28,10)
        self.fc1 = nn.Linear(10, 1)

    def forward(self, x):
        x = x.view(-1,28*28) 
        x = self.fc0(x) # 28*28 -> 30 
        x = F.relu(x) # Activation function
        x = self.fc1(x)  # 30 -> 10
        x = torch.sigmoid(x) # Activation function 
        return x
    

class netR(nn.Module):
    def __init__(self):
        super(netR, self).__init__() 
        # an affine operation: y = Wx + b
        self.fc0 = nn.Linear(28*28,10)
        self.fc1 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1,28*28) 
        x = self.fc0(x) # 28*28 -> 30 
        x = F.relu(x) # Activation function
        x = self.fc1(x)  # 30 -> 10
        return x


# 우선 u - net 구조로 짜보자..
class Conv_block(nn.Module):
    """반복되는 conv - BN - ReLU 구조 모듈화"""
    def __init__(self, in_channels, out_channels):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels = in_channels,
                       out_channels = out_channels,
                       kernel_size = 3,
                       padding= 1,
                       stride = 1)
        self.conv_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_bn(x)
        x = F.relu(x)
        return x


class netG(nn.Module):
    def __init__(self, input_channel):
        self.input_channel = input_channel
        
        super(netG, self).__init__()
        
        self.Conv_block1 = Conv_block(1,32)
        self.Conv_block2 = Conv_block(32,32)

        self.pool0 = nn.MaxPool2d(2) #14*14
        
        self.Conv_block3 = Conv_block(32,64)
        
        self.pool1 = nn.MaxPool2d(2) #7*7
        
        self.Conv_block3p5 = Conv_block(64,128)
        
        self.up0p5 = nn.Upsample(scale_factor=2, mode='bilinear', 
                               align_corners=True)#28*28
        
        self.Conv_block4 = Conv_block(128,64)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', 
                               align_corners=True)#28*28

        self.Conv_block5 = Conv_block(64,32)
        ##concat with conv0 feature map
        self.Conv_block6 = Conv_block(64,32)
        
        self.conv_last = nn.Conv2d(in_channels = 32,
                       out_channels = 1,
                       kernel_size = 3,
                       padding= 1,
                       stride = 1) # Layer 1

    def forward(self, x):
        noise = torch.randn(x.size()) * 0.1 # std 0.1
        
        x = self.Conv_block1(x)
        x4concat = self.Conv_block2(x)
        x = self.pool0(x4concat)
        x = self.Conv_block3(x)
        
        x = self.pool1(x)
        x = self.Conv_block3p5(x)
        x = self.up0p5(x)
        
        x = self.Conv_block4(x)
        x = self.up1(x)
        x = self.Conv_block5(x)
        x = torch.cat((x, x4concat), dim=1)
        x = self.Conv_block6(x)
        output = self.conv_last(x)
        
        return output


#trainer for Recongnition
class Trainer():
    def __init__(self, trainloader, testloader, net, optimizer, criterion):
        """
        trainloader: train data's loader
        testloader: test data's loader
        net: model to train
        optimizer: optimizer to update your model
        criterion: loss function
        """
        self.trainloader = trainloader
        self.testloader = testloader
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        
    def train(self, epoch = 1):
        """
        epoch: number of times each training sample is used
        """
        self.net.train()
        for e in range(epoch):
            running_loss = 0.0  
            for i, data in enumerate(self.trainloader, 0): 
                # get the inputs
                inputs, labels = data # Return type for data in dataloader is tuple of (input_data, labels)
                inputs = inputs.cuda()
                labels = labels.cuda()
                # zero the parameter gradients
                self.optimizer.zero_grad()    
                #  Question 1) what if we dind't clear up the gradients?

                # forward + backward + optimize
                outputs = self.net(inputs) # get output after passing through the network
                loss = self.criterion(outputs, labels) # compute model's score using the loss function 
                loss.backward() # perform back-propagation from the loss
                self.optimizer.step() # perform gradient descent with given optimizer

                # print statistics
                running_loss += loss.item()
                if (i+1) % 500 == 0:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (e + 1, i + 1, running_loss / 500))
                    running_loss = 0.0
        
        print('Finished Training')
        
    def test(self):
        self.net.eval() # Question 2) Why should we change the network into eval-mode?
    
        test_loss = 0
        correct = 0 # TP True Positives

        for inputs, labels in self.testloader:
            inputs = inputs.cuda()
            labels = labels.cuda() 
            output = self.net(inputs) 
            pred = output.max(1, keepdim=True)[1] # get the index of the max 
            correct += pred.eq(labels.view_as(pred)).sum().item()
            #TN += len(input) - pred.eq(labels.view_as(pred)).sum().item() #incorrect
            #print(pred.shape)
            test_loss /= len(self.testloader.dataset)
            #confusion matrix here
            #self.test_confusion_matrix = confusion_matrix(y_true = labels.view_as(pred).cpu(), y_pred = pred.cpu(),labels=[0,1,2,3,4,5,6,7,8,9])
        
        print('\nTest set:  Accuracy: {}/{} ({:.0f}%)\n'.
                format(correct, len(self.testloader.dataset),
                100.* correct / len(self.testloader.dataset)))
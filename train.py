from os import name
import torch 
import torch.nn as nn #
import torch.nn.functional as F # various activation functions for model
import torchvision # You can load various Pretrained Model from this package 
import torchvision.datasets as vision_dsets
import torchvision.transforms as T # Transformation functions to manipulate images
import torch.optim as optim # various optimization functions for model
from torch.autograd import Variable 
from torch.utils import data
from torchvision.utils import save_image
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

import model 
import dataset 
import utils 


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)
args = parser.parse_args()

recon_path, recon_pair_path, fake_path, fake_pair_path, test_fake_path, test_fake_pair_path = utils.file_generator(args.name)

mnist_train,mnist_test,trainDataLoader,testDataLoader= dataset.MNIST_DATA(batch_size = 32, download = True)  # Data Loader 
netD = model.netD().cuda() 
netG = model.netG(1).cuda() 
netR = model.netR().cuda()

#netR training
#netR = netR().cuda() # create the neural network instance and load to the cuda memory.
criterion = nn.CrossEntropyLoss() # Define Loss Function. We use Cross-Entropy loss.
optimizer = optim.SGD(netR.parameters(), lr=0.001) # optimizer receives training parameters and learning rate.

trainer = model.Trainer(trainloader = trainDataLoader,
                  testloader = testDataLoader,
                  net = netR ,
                  criterion = criterion,
                  optimizer = optimizer)
trainer.train(epoch = 4)
trainer.test()

#freeaze netR 
for para in netR.parameters():
    para.requires_grad = False

#netG reconsturction training

lr = 0.001
beta1 = 0.5

optimizerR = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
criterion = nn.L1Loss()

num_epochs = 1
for epoch in range(num_epochs):
    for i, data in enumerate(trainDataLoader, 0):
    
        netG.train()

        inputs, labels = data
        inputs = inputs.cuda()  
        netG.zero_grad()
        recon = netG(inputs)
        errR = criterion(recon, inputs)
        errR.backward()
        optimizerR.step()

        #print
        if i % 500 == 0:
            print(epoch,"-th epoch ",i , "-th iteration >> errD: ", errR.item())
        if i % 500 ==0:
            save_image(recon[0], recon_path+ '/recon_'+str(epoch)+"th_epoch_"+str(i)+'th_iter.png')
            save_image(inputs[0], recon_pair_path + '/real_'+str(epoch)+"th_epoch_"+str(i)+'th_iter.png')

#############################################
# setting for gan training 
criterionR = nn.CrossEntropyLoss()
criterionD = nn.BCELoss()
lr = 0.001
optimizerD = optim.Adam(netD.parameters(), lr=lr)
optimizerG = optim.Adam(netG.parameters(), lr=lr)
#optimizerR = optim.Adam(netG.parameters(), lr=lr)

# 통과한게 false
# 그냥 나온게 true
errD_list = []
errG_D_list = []
errG_R_list = []
count = 0
count_list = []

num_epochs = 1000
for epoch in range(num_epochs):
    for i, data in enumerate(trainDataLoader, 0):
        count = count + 1
        netD.train()
        netG.train()
        
        inputs, labels = data
        inputs4NetD = inputs.cuda()
        labels = labels.cuda()
        inputs4NetG = inputs4NetD.clone().detach().requires_grad_(True).cuda()
        #print(labels.shape)
        #print(labels.size(0))
        label_true = torch.full((labels.size(0),), 1 ,dtype=torch.float).cuda()
        label_false = torch.full((labels.size(0),), 0 ,dtype=torch.float).cuda()
        
        
        # Update D network
        netD.zero_grad()
        output = netD(inputs4NetD).view(-1)
        errD_real = criterionD(output, label_true)
     
        fake = netG(inputs4NetG)
        output = netD(fake).view(-1)
        errD_fake = criterionD(output, label_false)
        
        errD_total = errD_real + errD_fake
        errD_total.backward()
        optimizerD.step()

        #Update G network
        netG.zero_grad()
        output = netD(fake.detach()).view(-1)
        errG_D = criterionD(output, label_false)
    
        #Update R network
        netR.zero_grad()
        output = netR(fake.detach())
        errG_R = criterionR(output, labels)

        
        errG_total = errG_D + errG_R
        errG_total.backward()
        optimizerG.step()
        
        #print
        if count % 500 == 0:
            print(epoch,"-th epoch ",i,"-th iteration" )
            print("errD: ", errD_total.item()," errG_D: ",errG_D.item()," errG_R: ",errG_R.item())
            
            writer.add_scalar('Loss/errD', round(errD_total.item(), 5), count)
            writer.add_scalar('Loss/errG_D', round(errG_D.item(), 5), count)
            writer.add_scalar('Loss/errG_R', round(errG_R.item(), 5), count)

        if i % 500 ==0:
            save_image(fake[0], fake_path + '/fake_'+str(epoch)+"th_epoch_"+str(i)+'th_iter.png')
            save_image(inputs4NetD[0], fake_pair_path + '/real_'+str(epoch)+"th_epoch_"+str(i)+'th_iter.png')
        
        '''
        if epoch % 5 == 0 : #test
            count4test = 0
            netG.eval()
            for inputs, labels in testDataLoader:
                count4test = count4test + 1
                inputs = inputs.cuda()
                labels = labels.cuda()
                with torch.no_grad():
                    fake = netG(inputs).detach()
                save_image(fake[0], test_fake_path + '/fake_'+str(epoch)+"th_epoch_"+str(count4test)+'th_iter.png')
                save_image(inputs[0], test_fake_pair_path + '/real_'+str(epoch)+"th_epoch_"+str(count4test)+'th_iter.png')
        '''

writer.close()


'''        
if (epoch+1)%2 ==0:
    loss_plot(count_list ,errD_list, "errD_list")
    loss_plot(count_list ,errG_D_list, "errG_D_list")
    loss_plot(count_list ,errG_R_list, "errG_R_list")'''
            
            
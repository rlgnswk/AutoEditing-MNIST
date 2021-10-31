import torch 
import torch.nn as nn #
import torch.nn.functional as F # various activation functions for model
import torchvision
import torchvision.datasets as vision_dsets
from torch.utils import data
import torchvision.transforms as T

def MNIST_DATA(root=r'.',train =True,transforms=None ,download =True,batch_size = 32,num_worker = 1):
    print ("[+] Get the MNIST DATA")
    """
    We will use Mnist data for our tutorial 
    """
    mnist_train = vision_dsets.MNIST(root = root,  #root is the place to store your data. 
                                    train = True,  
                                    transform = T.ToTensor(),
                                    download=download)
    mnist_test = vision_dsets.MNIST(root = root,
                                    train = False, 
                                    transform = T.ToTensor(),
                                    download=download)
    """
    Data Loader is a iterator that fetches the data with the number of desired batch size. 
    * Practical Guide : What is the optimal batch size? 
      - Usually.., higher the batter. 
      - We recommend to use it as a multiple of 2 to efficiently utilize the gpu memory. (related to bit size)
    """
    trainDataLoader = data.DataLoader(dataset = mnist_train,  # information about your data type
                                      batch_size = batch_size, # batch size
                                      shuffle =True, # Whether to shuffle your data for every epoch. (Very important for training performance)
                                      num_workers =0) # number of workers to load your data. (usually number of cpu cores)

    testDataLoader = data.DataLoader(dataset = mnist_test, 
                                    batch_size = 128,
                                    shuffle = False, # we don't actually need to shuffle data for test
                                    num_workers = 0) #""
    print ("[+] Finished loading data & Preprocessing | train length: ", len(mnist_train)," test length: ",len(mnist_test))
    return mnist_train,mnist_test,trainDataLoader,testDataLoader

# If the download fails, you can try the following code. 
# !wget www.di.ens.fr/~lelarge/MNIST.tar.gz
# !tar -zxvf MNIST.tar.gz
# trainDset,testDset,trainDataLoader,testDataLoader= MNIST_DATA(batch_size = 32, download = True)  # Data Loader 

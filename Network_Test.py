import torch

import torch.nn as nn
import torch.optim as optim
#import torch.utils.data.dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torch import Tensor

import random

#load data
t_forms = transforms.Compose([transforms.Resize(28),transforms.ToTensor()])
batch_size=100

train_set = datasets.MNIST(root='./Data_MNIST',train=True, transform=t_forms, download=True)
test_set = datasets.MNIST(root='./Data_MNIST', train=False, transform=t_forms, download=True)
#dataset_size=len(train_set)
#print (dataset_size)

train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=True)
images_train,labels_train = next(iter(train_loader))
images_test,labels_test = next(iter(test_loader))
print(images_train.shape)
print(labels_train[10])
plt.imshow(images_train[10].reshape(28,28), cmap="gray")
plt.show()



#Define architecture elements

NN_type =dict()
NN_type['NN_type'] = ('CNN1d', 'CNN2d')

#solution space for layers

Space_layers = dict()
Space_layers['conv_layer'] = ('Conv1d','Conv2d', 'ConvTranspose1d', 'ConvTranspose2d')
Space_layers['pool_layer'] = ('MaxPool1d','MaxPool2d','AvgPool1d')
#Space_layers['pad_layer'] = ('')
Space_layers['linear'] =('Linear','Bilinear')
Space_layers['activ'] = ('relu','sigmoid','tanh','softmax','elu','linear')
Space_layers['norm'] = ('BatchNorm1d','GroupNorm')
Space_layers['num_neurons'] = (64,128,256,1024)
Space_layers['dropout_rate'] = (0.0, 0.2, 0.7)

#global parameters for network

Space_network = dict()
Space_network['num_layers'] = (1, 2, 3, 4)
Space_network['lr'] = (0.0001, 0.1, 0.15)
Space_network['weight_decay'] = (0.00001, 0.0004)
Space_network['optimizer'] = ('sgd', 'adam', 'adadelta', 'rmsprop')


#class Layer ()

#random value
def random_value(parameter):
    val = random.choice(parameter)
    return val

#create a NN
#random network

def random_network():

    global Space_layers_random_values, Space_network_random_values, network
    network = dict()

    Space_layers_random_values = dict()
    Space_network_random_values = dict()

    for i in Space_layers.keys():
        Space_layers_random_values.update({i: random_value(Space_layers[i])})
        network.update({i: random_value(Space_layers[i])})
 

    for i in Space_network.keys():
        Space_network_random_values.update({i: random_value(Space_network[i])})
        network.update({i: random_value(Space_network[i])})

    NN_layers = []
    values = network.values()
    NN_layers.append(values)
    return network

print(random_network())

#class Network ()
















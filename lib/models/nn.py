import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 10)
        #self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(81,10)
        #self.fc2 = nn.Linear(50,10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        #x = F.relu(self.conv1(x),2)
        kk = self.num_flat_features(x)
        #print kk
        x = x.view(-1,kk)
        
        #x = F.relu(self.fc1(x))
        x = self.fc1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def LLH( Model, q, idxs ):

    Vprior = 0
    Fprior = 0*q
    
    mm = (Model.net.parameters())
    
    vv = np.copy(q)

    for m in mm:
        nnn = np.prod(m.shape)
        kk = vv[:nnn]
        m.data   = torch.tensor( kk , dtype=torch.float64).view (m.shape).data.float()
        vv = vv[nnn:]
    
    
    
    Model.optimizer.zero_grad()
    
    inputs,labels = iter(Model.traindata).next()
    output = Model.net( inputs  )
    loss = Model.criterion(output, labels )
    loss.backward()
    
    V = loss.item()
    
    
    views = []

    mm = (Model.net.parameters())

    for m in mm:
        views.append( m.data.view(-1) )

    vv = torch.cat( views,0 )

    F = np.array(vv)
    F = np.reshape( F , q.shape )
    
    return np.log(V), F, Vprior, Fprior

    


def Setup( Model ):

    Model.net = Net()
    Model.criterion = nn.CrossEntropyLoss()
    Model.optimizer = optim.SGD(Model.net.parameters(), lr=0, momentum=0.9)

    print "Neural Network with the MNIST dataset "


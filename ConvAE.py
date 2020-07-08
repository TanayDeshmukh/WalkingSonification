#!/usr/bin/env python
# coding: utf-8

# In[1]:


#cnn autoencoder 
import torch.nn.functional as F
import torch
import torch.utils.data as data
import torch.nn as nn


# In[2]:


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 16), 3x3 kernels
        self.conv1 = nn.Conv1d(1, 32, 1)  
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv1d(32, 16, 1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool1d(1)
        
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose1d(16, 32, 1)
        self.t_conv2 = nn.ConvTranspose1d(32, 1, 1)


    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
            # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
                    ## decode ##
            # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv2(x))
                
        return x
    
    
model = ConvAutoencoder()
print(model)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.utils.data as data
from GaitSequenceDataset import GaitSequenceDataset
from preprocess import prepare_dataset,get_data_dimensions
from autoencoder import AE
from torch.nn import CrossEntropyLoss, MSELoss
from statistics import mean
from ConvAE import ConvAutoencoder

def train_model(model, dataset, lr, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # specify loss function
    criterion = nn.MSELoss()
    # specify loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_epochs = epochs

    for epoch in range(1, n_epochs+1):
        # monitor training loss
        train_loss = 0.0
        losses, seq_pred = [], []
        ###################
        # train the model #
        ###################
        for data in dataset:
            data=data.reshape((len(data),1,1))
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(data.float())
            #print(outputs)
            # calculate the loss
            loss = criterion(outputs, data.float())
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            #perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*data.size(0)
            
            losses.append(loss.item())
            seq_pred.append(outputs)

    return seq_pred,mean(losses)

def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t
 
def data_to_1d(data_set):
    t_list=[]
    len_data=len(data_set)
    for i in range(len_data):
        v=flatten(data_set[i])
        t=v.reshape(len(v),1)
        t_list.append(t)
    return t_list
    
def encoding_1d(train_dataset,lr,epoch,logging=False):
    train_set, seq_len, num_features = get_data_dimensions(train_dataset)
    train_1d= data_to_1d(train_set)
    train_data_1d,seq,num_of_features=get_data_dimensions(train_1d)
    data_1d = np.stack(train_data_1d)
    d_set=torch.from_numpy(data_1d)
    model = ConvAutoencoder()
    embeddings, f_loss = train_model(model, d_set, lr, epoch )

    return embeddings, f_loss

def main():
    lr=1e-3
    epochs=50

    dataset = GaitSequenceDataset(root_dir =r'C:\Users\yogit\Downloads\TUK\proj_HMMC\KL_Study_HDF5_for_learning\data',
                                longest_sequence = 85,
                                shortest_sequence = 55)

    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)
    train_dataset,test_dataset=prepare_dataset(dataloader)
    embeddings, f_loss = encoding_1d(train_dataset,lr=lr,epoch=epochs)
    print(f_loss)
    

if __name__ == "__main__":
    main()


# In[ ]:





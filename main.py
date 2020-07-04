import torch

import numpy as np
import torch.nn as nn
import torch.utils.data as data

from trainer import train, validate
from models import Encoder, Decoder, AutoEncoder
from torch.utils.tensorboard import SummaryWriter
from GaitSequenceDataset import GaitSequenceDataset
from torch.utils.data.sampler import SubsetRandomSampler

mode = 'train'
batch_size = 2
num_features = 6
embedding_dim = 512
sequence_length = 85
num_epochs = 10
learning_rate = 1e-4
load_pretrained = True

model_encoder_save_path = './model/model_encoder.pth'
model_decoder_save_path = './model/model_decoder.pth'
data_dir = '../../KL_Study_HDF5_for_learning/data/'

device = torch.device("cpu")

def generate_train_validation_samplers(dataset, validation_split):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(dataset_size * validation_split))
    np.random.shuffle(indices)
    train_indices, validation_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)

    return train_sampler, validation_sampler

def main():

    dataset = GaitSequenceDataset(root_dir = data_dir,
                                    longest_sequence = 85,
                                    shortest_sequence = 55)

    train_sampler, validation_sampler = generate_train_validation_samplers(dataset, validation_split=0.2)

    train_dataloader = data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_dataloader = data.DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler, drop_last=True)

    model = AutoEncoder(num_features, embedding_dim, sequence_length, batch_size, load_pretrained, 
                        model_encoder_save_path, model_decoder_save_path, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    if mode == 'train':
        summary = SummaryWriter()

        model.to(device)

        for epoch in range(num_epochs):

            # train(model, train_dataloader, optimizer, criterion, device, epoch, num_epochs, summary)

            if epoch % 2 == 0:
                validate(model, validation_dataloader, criterion, device, epoch, num_epochs, summary)

                encoder_state = {'epoch' : epoch,
                                'encoder' : model.encoder.state_dict()}
                decoder_state = {'epoch' : epoch,
                                'decoder' : model.decoder.state_dict()}

                torch.save(encoder_state, model_encoder_save_path)
                torch.save(decoder_state, model_decoder_save_path)
                break
    else:
        for batch_idx, sequences in enumerate(train_dataloader):
            sequences = sequences.permute(1, 0, 2)
            sequences.to(device)
            model.to(device)
            output = model(sequences)
            seq = sequences.permute(1, 0, 2)[0]
            out = output.permute(1, 0, 2)[0]
            np.savetxt('./seq1.csv', seq, delimiter=',')
            # for i in range(len(seq)):
            #     print(seq[i])
            #     print(out[i])
            #     print('-----------------------------------------------------')
            break

if __name__ == '__main__':
    main()
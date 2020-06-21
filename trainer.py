import sys
from IPython import display

def train(model, train_dataloader, optimizer, criterion, device, epoch, num_epochs, summary):
    model.train()
    num_batches = len(train_dataloader)
    for batch_idx, sequences in enumerate(train_dataloader):

        optimizer.zero_grad()
        sequences = sequences.permute(1, 0, 2)
        sequences.to(device)
        model.to(device)

        output = model(sequences)        
        loss = criterion(output, sequences)

        if(batch_idx % 10 == 0):
            display.clear_output(True)
            step = epoch * num_batches + batch_idx
            summary.add_scalar("Training Loss", loss.item(), step)
            print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(epoch,num_epochs, batch_idx, num_batches))
            print('Training Loss: {:.4f}'.format(loss))

        loss.backward()
        optimizer.step()


def validate(model, train_dataloader, criterion, device, epoch, num_epochs, summary):
    model.eval()
    num_batches = len(train_dataloader)
    for batch_idx, sequences in enumerate(train_dataloader):

        sequences = sequences.permute(1, 0, 2)
        sequences.to(device)
        model.to(device)

        output = model(sequences)        
        loss = criterion(output, sequences)

        if(batch_idx % 10 == 0):
            display.clear_output(True)
            step = epoch * num_batches + batch_idx
            summary.add_scalar("Validation Error", loss.item(), step)
            print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(epoch,num_epochs, batch_idx, num_batches))
            print('Validation Error: {:.4f}'.format(loss))


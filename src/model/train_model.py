import click
from dotenv import find_dotenv, load_dotenv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class BankDataset(Dataset):
    def __init__(self, data):
        self.all = torch.as_tensor(data)
        self.features = self.all[:, :-1]
        self.target = self.all[:, -1].reshape(-1, 1)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.target[idx]
        return x, y

class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 16)
        self.layer2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.Sigmoid()(x)
        x = self.layer2(x)
        x = nn.Sigmoid()(x)
        return x

@click.command()
@click.argument('dtrain_path', type=click.Path(exists=True))
@click.argument('dval_path', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path())

def main(dtrain_path, dval_path, model_path):
    """
    Train a PyTorch model using the training data and validation data that were 
    already processed earlier. These are saved to dtrain_path and dval_path as
    numpy arrays, where the rightmost column is the outcome variable. The 
    outcome variable is a binary variable.

    The PyTorch model has one hidden layer with 16 nodes. During training the 
    model performance is evaluated using the validation data, based on 
    the log-loss metric.

    The function writes out the best-performing model out to disk.
    """
    # Convert data to PyTorch Datasets
    dtrain = BankDataset(np.load(dtrain_path, allow_pickle=True))
    dval = BankDataset(np.load(dval_path, allow_pickle=True))

    # Create DataLoader for training and validation datasets
    batch_size = 32
    torch.manual_seed(1)
    train_loader = DataLoader(dtrain, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dval, batch_size=batch_size)

    # Initialize the model
    input_size = dtrain.features.shape[1]
    model = Model(input_size)

    # Define the loss function and optimizer
    learning_rate = 0.001
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = nn.BCELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop from the book
    best_val_loss = float('inf')
    num_epochs = 200
    log_epochs = 10
    train_loss = [0] * num_epochs
    val_loss = [0] * num_epochs

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss[epoch] += loss.item()*y_batch.size(0)
        train_loss[epoch] /= len(train_loader.dataset)

        # Validation
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                val_loss[epoch] +=  loss.item()*y_batch.size(0)
        # Calculate average validation loss for the epoch
        val_loss[epoch] /= len(val_loader.dataset)

        # Save model if it's the best performing
        if val_loss[epoch] < best_val_loss:
            best_val_loss = val_loss[epoch]
            torch.save(model, model_path)
        
        # Print training and validation loss after every `log_epochs`
        if epoch % log_epochs == 0:
            print(f"Epoch {epoch}: train loss {train_loss[epoch]:.3f} " + 
                  f"validation loss {val_loss[epoch]:.3f}")
    
if __name__ == '__main__':
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

import click
from dotenv import find_dotenv, load_dotenv
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


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
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.linear_stack(x)
        return x


def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    train_loss = 0.0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item() * y.size(0)

    train_loss /= len(dataloader.dataset)
    return train_loss


def val(dataloader, model, loss_fn, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item() * y.size(0)
    val_loss /= len(dataloader.dataset)
    return val_loss


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
    batch_size = 64
    torch.manual_seed(1)
    train_loader = DataLoader(dtrain, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dval, batch_size=batch_size)

    # Initialize the model
    input_size = dtrain.features.shape[1]
    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    model = Model(input_size).to(device)

    # Define the loss function and optimizer
    learning_rate = 0.001
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop from the book
    best_val_loss = float('inf')
    epochs = 200
    log_epochs = 10
    for t in range(epochs):
        train_loss = train(train_loader, model, loss_fn, optimizer, device)
        val_loss = val(val_loader, model, loss_fn, device)
        if (t + 1) % log_epochs == 0:
            print(f"Epoch {t+1}\n-------------------------------")
            print(f"Train loss: {train_loss:.5f}\n" +
                  f"Validation loss: {val_loss:.5f}")
        # Save model if it's the best performing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, model_path)


if __name__ == '__main__':
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

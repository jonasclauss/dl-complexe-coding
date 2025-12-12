from helpers.dataset import DataReader
from helpers.nn import NeuralNetwork
from torch import nn
from torch.utils.data import DataLoader
import torch

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':
    print("Deep learning project by Jonas, Robin and Lukas.")

    dr = DataReader(data_path='./data/EuroSAT_RGB')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"You re using {device} :)")
    model = NeuralNetwork().to(device)

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(dr.training_set, batch_size=batch_size)
    test_dataloader = DataLoader(dr.test_set, batch_size=batch_size)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")
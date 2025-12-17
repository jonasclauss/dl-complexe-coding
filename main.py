import os
from helpers.dataset import DataReader
from helpers.nn import NeuralNetwork, PretrainedResNet18
from torch import inf, nn
from torch.utils.data import DataLoader
import torch
from alive_progress import alive_bar
from torchvision import transforms
from matplotlib import pyplot as plt

train_loss_array = []
test_loss_array = []
best_loss = inf

def train(dataloader, model, loss_fn, optimizer):
    global train_loss_array
    size = len(dataloader)
    model.train()
    with alive_bar(size, title='Training', bar='bubbles') as bar:
        loss_arr = []
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred: torch.Tensor = model(X)
            y = y.long().to(device)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_arr.append(loss.item()) 
            #loss, current = loss.item(), (batch + 1) * len(X)
            bar()
        train_loss_array.append(sum(loss_arr)/len(loss_arr))
        

def test(dataloader, model, loss_fn):
    global best_loss, test_loss_array
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y = y.long().to(device)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    test_loss_array.append(test_loss)
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    if best_loss > test_loss:
        torch.save(model.state_dict(), 'model.pth')
        best_loss = test_loss

if __name__ == '__main__':
    print("Deep learning project by Jonas, Robin and Lukas.")

    #transform from 64x64 to 224x224 for ResNet18
    #transform = transforms.Compose([
     #   transforms.Resize((224, 224)),
      #  transforms.ToTensor(),
       # transforms.Normalize(
        #    mean=[0.485, 0.456, 0.406],
         #   std=[0.229, 0.224, 0.225],
        #),
    #])


    dr = DataReader(data_path='./data/EuroSAT_RGB')#, transform=transform)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"You re using {device} :)")
    model = NeuralNetwork().to(device)
    #model = PretrainedResNet18(num_classes=10, freeze_backbone=False).to(device)

    loss_fn = nn.CrossEntropyLoss()
    epochs = 20
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


    batch_size = 64
    # Create data loaders.
    train_dataloader = DataLoader(dr.training_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dr.test_set, batch_size=batch_size, shuffle=True)

    #if os.path.exists('model.pth'):
    #        model = NeuralNetwork().to(device)
    #        model.load_state_dict(torch.load('model.pth'))
        
    

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    plt.plot(range(1, epochs+1), train_loss_array, label='Training Loss')
    plt.plot(range(1, epochs+1), test_loss_array, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss over Epochs')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()
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
NUM_CLASSES = 10 

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
        

def evaluate(dataloader, model, loss_fn, save_best = False):
    global best_loss, test_loss_array
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y = y.long().to(device)

            test_loss += loss_fn(pred, y).item()

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            all_preds.append(pred.argmax(1).cpu())
            all_labels.append(y.cpu())
            
    test_loss /= num_batches
    correct /= size
    conf = build_confusion_matrix(torch.cat(all_labels), torch.cat(all_preds), max(len(torch.unique(torch.cat(all_labels))), len(torch.unique(torch.cat(all_preds)))))
    
    print(conf)
    if save_best == True: test_loss_array.append(test_loss)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    if save_best and best_loss > test_loss:
        torch.save(model.state_dict(), 'model.pth')
        best_loss = test_loss



def build_confusion_matrix(labels: torch.Tensor,
                           preds: torch.Tensor,
                           num_classes: int) -> torch.Tensor:
    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(labels, preds):
        conf[t, p] += 1
    return conf

def set_fixed_seed(matno):
    torch.manual_seed(matno)
    torch.cuda.manual_seed_all(matno)

if __name__ == '__main__':
    print("Deep learning project by Jonas, Robin and Lukas.")
    MATRIKENUMMER = 3792567
    set_fixed_seed(MATRIKENUMMER)

    train_transform_mild = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5)
    ])

    train_transform_strong = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.RandomResizedCrop(size=64, scale=(0.8, 1.0)),
    # transforms.ColorJitter(
    #     brightness=0.2,
    #     contrast=0.2,
    #     saturation=0.2,
    #     hue=0.05
    # ),
    ])


    dr = DataReader(data_path='./data/EuroSAT_RGB', transform=train_transform_strong)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"You re using {device} :)")
    model = NeuralNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()
    epochs = 30
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


    batch_size = 128
    # Create data loaders.
    train_dataloader = DataLoader(dr.training_set, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True)
    val_dataloader = DataLoader(dr.validation_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dr.test_set, batch_size=batch_size, shuffle=True)

        
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        
        train(train_dataloader, model, loss_fn, optimizer)
        evaluate(val_dataloader, model, loss_fn, save_best=True)
        model_test = NeuralNetwork().to(device)
        model_test.load_state_dict(torch.load('model.pth'))
    print("Done!")

    #testing stuff
    evaluate(test_dataloader, model_test, loss_fn, save_best=False)

    plt.plot(range(1, epochs+1), train_loss_array, label='Training Loss')
    plt.plot(range(1, epochs+1), test_loss_array, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss over Epochs')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()
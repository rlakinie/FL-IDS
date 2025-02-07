import torch
import torch.nn as nn

from datasets.utils.logging import disable_progress_bar

import flwr as fl

from tqdm import tqdm

from client_data import load
#from memory_profiler import profile
import time
 
# record start time


DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
#print(
    #f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
#)
disable_progress_bar()

NUM_CLIENTS = 3
#BATCH_SIZE = 32






class  conv_net(nn.Module):
    def __init__(self):
        super(conv_net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Flatten(),
            nn.Linear(384, 512), # Adjust the input features to 384
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )

        self.classification_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.classification_head(x)

        return x
    

def train(net, trainloader, optimizer, epochs, device):
    """Train the model on the training set."""
    criterion = torch.nn.BCELoss()
    net.train()
    for epoch in range(epochs):
        for features, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(features.to(device)), labels.to(device)).backward()
            optimizer.step()


def evaluate(model, test_loader, device):
    model.eval()
    criterion = nn.BCELoss()
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader):
            outputs = model(X_batch.to(device))
            batch_loss = criterion(outputs, y_batch.to(device))
            loss += batch_loss.item()
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = correct / total
    loss = loss / len(test_loader)
    return loss, accuracy

@profile
def main():
    start = time.time()
    trainset, valset = load(0, NUM_CLIENTS)
    net = conv_net().to(DEVICE)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    train(net, trainset, optimizer, 15, DEVICE)
    loss, accuracy = evaluate(net, valset, DEVICE)
    print(f" validation loss {loss}, accuracy {accuracy}")
    end = time.time()
    print("The time of execution of above program is :",(end-start), "s")


if __name__ == "__main__":
    main()

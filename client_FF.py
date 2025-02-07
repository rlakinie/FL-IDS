import argparse
import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from client_data import load
#from memory_profiler import profile


parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="0.0.0.0:8080",
    help=f"gRPC server address (default '0.0.0.0:8080')",
)
parser.add_argument(
    "--cid",
    type=int,
    required=True,
    help="Client id. Should be an integer between 0 and NUM_CLIENTS",
)
'''parser.add_argument(
    "--num_clients",
    type=int,
    action="number of clients",
    #required=True,
    help="number of clients must be greater than client id",
)
'''
warnings.filterwarnings("ignore", category=UserWarning)

NUM_CLIENTS = 3




# The following lines define a simple convolutional neural network

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
        #x = x.view(x.size(0), -1)
        x = self.classification_head(x)

        return x
    


def train(net, trainloader, optimizer, epochs, device):
    """Train the model on the training set."""
    # Freeze all the layers of the pre-trained model
    #for param in net.parameters():
     #   param.requires_grad = False

    # Modify the model's head for a new task

    net.requires_grad_(False)
    # Now enable just for output head
    net.classification_head.requires_grad_(True)


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




# Flower client, adapted from Pytorch quickstart/simulation example
class FlowerClient(fl.client.NumPyClient):
    """A FlowerClient that trains a MobileNetV3 model for CIFAR-10 or a much smaller CNN
    for MNIST."""

    def __init__(self, trainset, valset):
        self.trainset = trainset
        self.valset = valset
        self.model = conv_net()
        # Determine device
        self.device = torch.device("cpu")
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

    def set_parameters(self, params):
        """Set model weights from a list of NumPy ndarrays."""
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict(
            {
                k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
                for k, v in params_dict
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    @profile
    def fit(self, parameters, config):
        print("Client sampled for fit()")
        self.set_parameters(parameters)
        # Read hyperparameters from config set by the server
        epochs = config["epochs"]
        # Construct dataloader
        # Define optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        # Train
        train(self.model, self.trainset, optimizer, epochs=epochs, device=self.device)
        # Return local model and statistics
        return self.get_parameters({}), len(self.trainset.dataset), {}

    def evaluate(self, parameters, config):
        print("Client sampled for evaluate()")
        self.set_parameters(parameters)

        # Evaluate
        loss, accuracy = evaluate(self.model, self.valset, device=self.device)

        # Return statistics
        return float(loss), len(self.valset.dataset), {"accuracy": float(accuracy)}



def main():
    args = parser.parse_args()
    print(args)

    assert args.cid < NUM_CLIENTS


    # Download dataset and partition it
    trainset, valset = load(args.cid, NUM_CLIENTS)

    # Start Flower client setting its associated data partition
    fl.client.start_client(
        server_address=args.server_address,
        client=FlowerClient(
            trainset=trainset, valset=valset
        ).to_client(),
    )


if __name__ == "__main__":
    main()

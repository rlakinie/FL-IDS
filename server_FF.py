import argparse
from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics

import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm

from serverFF_data_proxy import load

NUM_SERVER = 1


parser = argparse.ArgumentParser(description="Flower Embedded devices")

parser.add_argument(
    "--server_address",
    type=str,
    default="0.0.0.0:8080",
    help=f"gRPC server address (deafault '0.0.0.0:8080')",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=5,
    help="Number of rounds of federated learning (default: 5)",
)
parser.add_argument(
    "--sample_fraction",
    type=float,
    default=1.0,
    help="Fraction of available clients used for fit/evaluate (default: 1.0)",
)
parser.add_argument(
    "--min_num_clients",
    type=int,
    default=3,
    help="Minimum number of available clients required for sampling (default: 2)",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size for server training data"
)

parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    help="epochs for server training data"
)

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


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """This function averages teh `accuracy` metric sent by the clients in a `evaluate`
    stage (i.e. clients received the global model and evaluate it on their local
    validation sets)."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_config(server_round: int):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 5,  # Number of local epochs done by clients
    }
    return config


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]



def server_train(trainloader, DEVICE, net, epochs):
    # Define optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    train(net, trainloader, optimizer, epochs, DEVICE)
    params = get_parameters(net)
    return params



def main():

    args = parser.parse_args()
    print(args)

    
    epochs = args.epochs

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Try "cuda" to train on GPU
    print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")
    net = conv_net()
    net.to(DEVICE)

    trainset, valset = load(0, NUM_SERVER)


    params = server_train(trainset, DEVICE, net, epochs)


    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.sample_fraction,
        fraction_evaluate=args.sample_fraction,
        min_fit_clients=args.min_num_clients,
        on_fit_config_fn=fit_config,
        min_evaluate_clients=args.min_num_clients,  # Never sample less than 5 clients for evaluation
        min_available_clients=args.min_num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=fl.common.ndarrays_to_parameters(params)
    )

    #dp_strategy = fl.server.strategy.DifferentialPrivacyServerSideFixedClipping(strategy, 1, )

    # Start Flower server
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()

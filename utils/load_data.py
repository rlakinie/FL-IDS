#Imports

import argparse
import flwr as fl
from datasets import load_dataset
from flwr_datasets.partitioner import FederatedDataset
from datasets.utils.logging import disable_progress_bar

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn import preprocessing

from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

NUM_CLIENTS = 10

parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--cat_columns",
    type=object,
    default=['is_host_login', 'protocol_type', 'service', 'flag', 'land', 'logged_in','is_guest_login', 'level', 'class'],
    help=f"Categorical columns for each dataset",
)

parser.add_argument(
    "--train_dataset",
    type=str,
    default="nsl",
    help=f"Train dataset",
    )

parser.add_argument(
    "--test_dataset",
    type=str,
    default="nsl",
    help=f"Test dataset",
    )


#nsl = ['is_host_login', 'protocol_type', 'service', 'flag', 'land', 'logged_in','is_guest_login', 'level', 'class']


def retrieve(*args):
    # retrieve data from filepath
    if args.train_dataset == 'nsl':
        data = pd.read_csv("/home/student/embedded-devices/play/data/KDDTrain+.txt")
    return data

def scaling(data, columns):
    scaler = RobustScaler()
    scalar_fit_transform = scaler.fit_transform(data)
    return pd.DataFrame(scalar_fit_transform, columns)


def LE(data):
    #Label encoding. Try LE vs get_dummies and see performance
    for column in data.columns:
        if data[column].dtype == type(object):
            le = preprocessing.LabelEncoder()
            data[column] = le.fit_transform(data[column])
    return data



def feature_engineering(data, *args):

    #Data prerocessing: might involve label encoding, column removal, scaling. Rest would require me digging into the data
    #function to determine best features to select

    #Selecting numerical columns to perform scaling on, replacing old numerical column
    columns = args.cat_columns

    feat = data.drop(columns)
    feat_columns = feat.columns
    scaled = scaling(feat, feat_columns)
    data.drop(labels=feat_columns, axis="columns", inplace=True)
    data[feat_columns] = scaled[feat_columns]
    data = LE(data)

    data.loc[data['outcome'] == "normal", "outcome"] = 0
    data.loc[data['outcome'] != 0, "outcome"] = 1

    return data

def data_split(data):
    Y = data.iloc[:,-1].to_numpy()
    X = data.drop(columns=data.columnss[-1]).to_numpy()
    return X, Y


def get_data(data, *args):
    train = retrieve(args.train_dataset)
    test = retrieve(args.test_dataset)

    train = feature_engineering(train, args.cat_columns)
    test = feature_engineering(test, args.cat_columns)
    
    X_train, Y_train = data_split(train)
    X_test, Y_test = data_split(test)

    return X_train, Y_train, X_test, Y_test


    


def load_datasets():
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})

   
    # Create train/val for each partition and wrap it into DataLoader
    trainloaders = []
    valloaders = []
    for partition_id in range(NUM_CLIENTS):
        partition = fds.load_partition(partition_id, "train")
        partition = partition.train_test_split(train_size=0.8, seed=42)
        trainloaders.append(DataLoader(partition["train"], batch_size=BATCH_SIZE))
        valloaders.append(DataLoader(partition["test"], batch_size=BATCH_SIZE))
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloaders, valloaders, testloader


trainloaders, valloaders, testloader = load_datasets()



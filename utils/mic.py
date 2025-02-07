#Imports

import argparse
import warnings

import flwr as fl
from flwr_datasets.partitioner import IidPartitioner

from datasets import Dataset, load_dataset
from datasets.utils.logging import disable_progress_bar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, OrdinalEncoder

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

#Constants
#NUM_CLIENTS = 4

def retrieve(file_path):
    # retrieve data from filepath
    #if args.train_dataset == 'nsl':
    columns = ([ 'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations','num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate','rerror_rate', 'srv_rerror_rate' ,'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 
    'dst_host_count','dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class', 'level'])
    data = pd.read_csv(file_path, names = columns)
    return data

def load(partition_id: int, NUM_CLIENTS: int):

    dataset = Dataset.from_pandas(retrieve("data/KDDTest-21.txt"))
    partitioner = IidPartitioner(NUM_CLIENTS)
    partitioner.dataset = dataset
    data_ = partitioner.load_partition(partition_id).with_format("pandas")[:]

    X = data_.drop("class", axis = 1)
    y = data_["class"].apply(lambda x: 0 if x=="normal" else 1)


    cat_features = X.select_dtypes(include=["object"]).columns
    X[cat_features] = OrdinalEncoder().fit_transform(X[cat_features])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    num_features = X.select_dtypes(include=["float64", "int64"]).columns
    scaling = ColumnTransformer(transformers=[("num", StandardScaler(), num_features)])

    X_train_scale = scaling.fit_transform(X_train)
    X_val_scale = scaling.transform(X_val)



    pca = PCA(n_components = 3)
    X_train, X_val = pca.fit_transform(X_train_scale), pca.transform(X_val_scale)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=32, shuffle=False)


    return trainloader, valloader
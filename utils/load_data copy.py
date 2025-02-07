#Imports

import argparse
import flwr as fl
from datasets import load_dataset
from flwr_datasets.partitioner import IidPartitioner
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



def retrieve(file_path):
    # retrieve data from filepath
    #if args.train_dataset == 'nsl':
    columns = (['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes','land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations','num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate','rerror_rate', 'srv_rerror_rate' ,'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 
    'dst_host_count','dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class', 'level'])
    data = pd.read_csv(file_path, names = columns)
    
    target = data['class']
    data.drop(columns = ['class'], inplace = True)
    data['class'] = target
    return data




def scaling(data, columns):
    scaler = RobustScaler()
    scalar_fit_transform = scaler.fit_transform(data)
    return pd.DataFrame(scalar_fit_transform, columns = columns)



def LE(data):
    #Label encoding. Try LE vs get_dummies and see performance
    for column in data.columns:
        if data[column].dtype == type(object):
            le = preprocessing.LabelEncoder()
            data[column] = le.fit_transform(data[column])
    return data



def feature_engineering(data):

    #Data prerocessing: might involve label encoding, column removal, scaling. Rest would require me digging into the data
    #function to determine best features to select

    #Selecting numerical columns to perform scaling on, replacing old numerical column
    nsl = ['is_host_login', 'protocol_type', 'service', 'flag', 'land', 'logged_in','is_guest_login', 'level', 'class']
    columns = nsl

    feat = data.drop(columns = columns)
    feat_columns = feat.columns
    scaled = scaling(feat, feat_columns)
    data.drop(labels=feat_columns, axis="columns", inplace=True)
    data[feat_columns] = scaled[feat_columns]
    data = LE(data)

    data.loc[data['class'] == "normal", "class"] = 0
    data.loc[data['class'] != 0, "class"] = 1

    return data



def data_split(data):
    Y = data['class'].to_numpy()
    X = data.drop(columns='class').to_numpy()
    return X, Y


def get_data():
    train = retrieve("/content/KDDTrain+.txt")
    test = retrieve("/content/KDDTest+.txt")

    train = feature_engineering(train)
    test = feature_engineering(test)
    
    X_train, Y_train = data_split(train)
    X_test, Y_test = data_split(test)

    return X_train, Y_train, X_test, Y_test


    


def load_datasets(num_clients):

    
    xtrain = retrieve("/content/KDDTrain+.txt")
    xtest = retrieve("/content/KDDTest+.txt")

    ftrain = feature_engineering(xtrain)
    ftest = feature_engineering(xtest)
    

    # ftrainx, ftrainy, _, _ = get_data()
    # Create train/val for each partition and wrap it into DataLoader
    trainloaders = []
    valloaders = []
    partitioner = IidPartitioner({"train": num_clients})
    partitioner.dataset = ftrain
    for partition_id in range(NUM_CLIENTS):
        partition = partitioner.load_partition(partition_id, "train")

        partition = partition.train_test_split(train_size=0.8, seed=42)


        trainloaders.append(DataLoader(partition["train"], batch_size=BATCH_SIZE))
        valloaders.append(DataLoader(partition["test"], batch_size=BATCH_SIZE))
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloaders, valloaders, testloader


trainloaders, valloaders, testloader = load_datasets()



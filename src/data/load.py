import torch
import torchvision
import pandas as pd
from torch.utils.data import TensorDataset
from sklearn import datasets
from sklearn.model_selection import train_test_split
# Testing
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

def load(train_size=.8):
    
    # Load data
    housing = datasets.fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = housing.target
    X, y = X[::2], y[::2]  # subsample for faster demo
    wandb.errors.term._show_warnings = False
    # ignore warnings about charts being built from subset of data

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
    """
    # Load the data
    """
    # the data, split between train and test sets
    # train = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    # test = torchvision.datasets.MNIST(root='./data', train=False, download=True)

    # (x_train, y_train), (x_test, y_test) = (train.data, train.targets), (test.data, test.targets)

    # split off a validation set for hyperparameter tuning
    # x_train, x_val = x_train[:int(len(train)*train_size)], x_train[int(len(train)*train_size):]
    # y_train, y_val = y_train[:int(len(train)*train_size)], y_train[int(len(train)*train_size):]

    training_set = TensorDataset(X_train, y_train)
    test_set = TensorDataset(X_test, y_test)
    _datasets = [training_set, test_set]
    return _datasets

def load_and_log():
    # 🚀 start a run, with a type to label it and a project it can call home
    with wandb.init(
        project="yml_2_trabajo",
        name=f"Load Raw Data ExecId-{args.IdExecution}", job_type="load-data") as run:
        
        _datasets = load()  # separate code for loading the datasets
        names = ["training", "test"]

        # 🏺 create our Artifact
        raw_data = wandb.Artifact(
            "California-housing-raw", type="dataset",
            description="raw California-housing dataset, split into train/test",
            metadata={"source": "sklearn.datasets.fetch_california_housing",
                      "sizes": [len(dataset) for dataset in _datasets]})

        for name, data in zip(names, _datasets):
            # 🐣 Store a new file in the artifact, and write something into its contents.
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        # ✍️ Save the artifact to W&B.
        run.log_artifact(raw_data)

# testing
load_and_log()
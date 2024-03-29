import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
# Testing
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

def load():
    
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
    _datasets = [X_train, X_test, y_train, y_test]
    return _datasets

def load_and_log():
    # 🚀 start a run, with a type to label it and a project it can call home
    with wandb.init(
        project="yml_2_trabajo",
        name=f"Load Raw Data ExecId-{args.IdExecution}", job_type="load-data") as run:
        
        _datasets = load()  # separate code for loading the datasets
        names = ["X_train", "X_test", "y_train", "y_test"]

        # 🏺 create our Artifact
        raw_data = wandb.Artifact(
            "California-housing-raw", type="dataset",
            description="raw California-housing dataset, split into train/test",
            metadata={"source": "sklearn.datasets.fetch_california_housing",
                      "sizes": [len(dataset) for dataset in _datasets]})

        for name, data in zip(names, _datasets):
            # 🐣 Store a new file in the artifact, and write something into its contents.
            with raw_data.new_file(name + ".npy", mode="wb") as file:
               
                np.save(file, data)

        # ✍️ Save the artifact to W&B.
        run.log_artifact(raw_data)

# testing
load_and_log()
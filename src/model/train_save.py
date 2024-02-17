import torch
import torchvision
import joblib
import numpy as np
from torch.utils.data import TensorDataset
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#testing
import os
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

def train_save_and_log(steps):

    with wandb.init(
        project="yml_2_trabajo",
        name=f"Train and save Model ExecId-{args.IdExecution}", 
        job_type="train and save-model") as run:    
        
        model_artifact = wandb.Artifact(
            "regression-trained-model", type="model",
            description="Regression trained model",
            metadata=steps)
         
        # ✔️ Load dataset
        raw_data_artifact = run.use_artifact('california-housing-raw:latest')

        # 📥 Extract dataset
        raw_dataset = raw_data_artifact.download(root="./data/artifacts/")
        X_train=read(raw_dataset, "X_train")
        X_test=read(raw_dataset, "X_test")
        y_train=read(raw_dataset, "y_train")
        y_test=read(raw_dataset, "y_test")
        
        # Train model, get predictions
        reg = Ridge()
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        run.summary.update({"MSE": mse, "R2": r2})
        joblib.dump(reg, 'reg_model.pkl')
        model_artifact.add_file("reg_model.pkl")
        wandb.save("reg_model.pkl")
        run.log_artifact(model_artifact)

def read(data_dir, split):
    filename = split + ".npy"
    x = np.load(os.path.join(data_dir, filename))
    return x

steps = {"Train": True,
         "Save": True}

train_save_and_log(steps)
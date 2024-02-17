import torch
import torchvision
import joblib
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
else:
    args.IdExecution = "testing console"

def preprocess(dataset, normalize=True, expand_dims=True):
    """
    ## Prepare the data
    """
    x, y = dataset.tensors

    if normalize:
        # Scale images to the [0, 1] range
        x = x.type(torch.float32) / 255

    if expand_dims:
        # Make sure images have shape (1, 28, 28)
        x = torch.unsqueeze(x, 1)
    
    return TensorDataset(x, y)

def preprocess_and_log(steps):

    with wandb.init(project="yml_2_trabajo",name=f"Train and save Model ExecId-{args.IdExecution}", job_type="train and save-model") as run:    
        processed_data = wandb.Artifact(
            "regression-model", type="model",
            description="Regression model",
            metadata=steps)
         
        # ✔️ declare which artifact we'll be using
        raw_data_artifact = run.use_artifact('california-housing-raw:latest')

        # 📥 if need be, download the artifact
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
        joblib.dump(reg, './data/model/reg_model.pkl')
        run.log_artifact(processed_data)

def read(data_dir, split):
    filename = split + ".pt"
    x, y = torch.load(os.path.join(data_dir, filename))

    return TensorDataset(x, y)

steps = {"normalize": True,
         "expand_dims": False}

preprocess_and_log(steps)
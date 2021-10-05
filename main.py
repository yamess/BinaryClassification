import os
import pickle

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import lr_scheduler

from src.config import config
from src.engine import engine
from src.train import predict_proba, validate
from src.model import NeuralNet
from src.utils import init_random_seed, get_cat_emb_dims
from src.dataset import DataTransformer, CustomDataset
from torch import optim
import torch.nn as nn
import torch
import pandas as pd

if __name__ == "__main__":
    # environment setup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    init_random_seed(seed=config["SEED"])

    trainDF = pd.read_csv(config["TRAIN_DATA_PATH"], delimiter=";")
    data_transformer = DataTransformer(is_train_mode=True)
    processed_df = data_transformer.transform(trainDF)

    # Save the encoders and the scaler for later use
    with open("models/encoders.pickle", "wb") as f:
        pickle.dump(data_transformer.encoders, f)
    with open("models/scaler.pickle", "wb") as f:
        pickle.dump(data_transformer.scaler, f)

    # embedding columns and dimensions: cat column with more than 2 categories
    emb_dims, emb_cols = get_cat_emb_dims(
        cat_data=processed_df.loc[:, data_transformer.cat_cols]
    )
    cont_cols = [
        c for c in processed_df.columns if c not in emb_cols and c != "install" and c != "id"
    ]

    y = processed_df.install
    X = processed_df.drop("install", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.85, random_state=config["SEED"], stratify=y
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, train_size=0.85, random_state=config["SEED"], stratify=y_train
    )

    # Create dataset instance to pass to dataloader
    # Training set
    train_dataset = CustomDataset(
        emb_cols=emb_cols, cont_cols=cont_cols, x=X_train, y=y_train
    )

    # Validation set
    valid_dataset = CustomDataset(
        emb_cols=emb_cols, cont_cols=cont_cols, x=X_valid, y=y_valid
    )

    # Dataloader
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=config["TRAIN_BS"],
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        sampler=SequentialSampler(valid_dataset),
        batch_size=config["VALID_BS"],
    )

    # Instanciate the model
    model = NeuralNet(cont_input_size=len(cont_cols), emb_dims=emb_dims)
    model.to(config["DEVICE"])

    checkpoint = None
    if os.path.isfile(config["CHECKPOINT_PATH"]):
        checkpoint = torch.load(config["CHECKPOINT_PATH"])
        model.load_state_dict(checkpoint["best_state_dict"])

    # Instanciate the optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["LR"])

    # we're setting the positive class weight because the data is imbalanced
    pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    criterion.to(config["DEVICE"])

    # The scheduler for the learning rate adjustment
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=10,
    )

    engine(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=valid_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        config=config,
        checkpoint=checkpoint
    )

    # Compute the model performance on the test set

    checkpoint = torch.load("models/checkpoint.pt")
    state_dict = checkpoint["best_state_dict"]
    model.load_state_dict(state_dict)

    # Create the dataloader for the test set
    test_dataset = CustomDataset(
        emb_cols=emb_cols, cont_cols=cont_cols, x=X_test, y=y_test
    )

    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=2000,
    )
    performance, _ = validate(model=model, dataloader=test_dataloader, criterion=criterion, device=config["DEVICE"])
    print(performance)

    # Predict the install probabilities
    PRED_DF_PATH = "data/assessment_data.csv"
    pred_df = pd.read_csv(PRED_DF_PATH, delimiter=";")

    with open("models/encoders.pickle", "rb") as f:
        encoder = pickle.load(f)

    with open("models/scaler.pickle", "rb") as f:
        scaler = pickle.load(f)

    test_data_transformer = DataTransformer(encoders=encoder, scaler=scaler, is_train_mode=False)
    X_assessment = test_data_transformer.transform(pred_df)

    pred_dataset = CustomDataset(
        emb_cols=emb_cols, cont_cols=cont_cols, x=X_assessment, y=None
    )
    pred_dataloader = DataLoader(
        pred_dataset,
        sampler=SequentialSampler(pred_dataset),
        batch_size=2000,
    )
    predictions = predict_proba(model=model, dataloader=pred_dataloader, device=config["DEVICE"])
    pred = [p[0] for p in predictions]
    data = pd.DataFrame(list(zip(X_assessment.id, pred)), columns=["ID", "install_proba"])

    # save the predicted probability as a csv file
    data.to_csv("install_proba.csv", index=False)

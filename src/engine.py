import csv
from copy import deepcopy
import time

import torch

from src.train import train, validate


def engine(
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    criterion,
    checkpoint,
    config,
    scheduler=None,
):

    engine_start_time = time.time()
    if checkpoint:
        best_mbe = checkpoint["best_mbe"]
        best_roc_auc = checkpoint["best_roc_auc"]
        epoch_at_best = checkpoint["epoch_at_best"]
    else:
        checkpoint = {}
        best_mbe = 1e10
        best_roc_auc = 0.0
        epoch_at_best = 0

    print("======================= Training Started ============================")

    for e in range(1 + epoch_at_best, config["N_EPOCHS"] + epoch_at_best):
        e_start_time = time.time()

        metrics_train = train(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=config["DEVICE"]
        )
        metrics_valid, _ = validate(
            model=model,
            dataloader=eval_dataloader,
            criterion=criterion,
            device=config["DEVICE"]
        )

        if scheduler is not None:
            scheduler.step(metrics_valid["loss"])

        e_end_time = time.time()
        e_elapsed_time = e_end_time - e_start_time

        display_msg = (
            f"Epoch: {e: <{4}} | Elapsed Time: {e_elapsed_time: 3.2f} s | Train Loss: {metrics_train['loss']: .4f} | "
            f"Valid Loss: {metrics_valid['loss']: .4f} | Train ROC AUC: {metrics_train['roc_auc']: .4f} | "
            f"Valid ROC AUC: {metrics_valid['roc_auc']: .4f} | Train MBE: {metrics_train['mbe']: .4f} | "
            f"Valid MBE: {metrics_valid['mbe']: .4f} | "
        )

        if metrics_valid["roc_auc"] > best_roc_auc and metrics_valid["mbe"] < best_mbe:
            best_valid_loss = metrics_valid["loss"]
            best_roc_auc = metrics_valid["roc_auc"]
            best_mbe = metrics_valid["mbe"]
            best_state_dict = deepcopy(model.state_dict())

            display_msg += " + "

            checkpoint["epoch_at_best"] = e
            checkpoint["best_valid_loss"] = best_valid_loss
            checkpoint["best_roc_auc"] = best_roc_auc
            checkpoint["best_mbe"] = best_mbe
            checkpoint["best_state_dict"] = best_state_dict

            torch.save(checkpoint, config["CHECKPOINT_PATH"])

        print(display_msg)

    engine_end_time = time.time()
    total_time = engine_end_time - engine_start_time
    print(f"Total Time elapsed: {total_time: .4f}")
    print("======================== End of Training ===================")
    print(" *********************** SUMMARY FOR VALIDATION ***********************")
    print(f"  Best Model loss: {checkpoint['best_valid_loss']}")
    print(f"  Best Model ROC AUC: {checkpoint['best_roc_auc']}")
    print(f"  Best Model Prediction Bias: {checkpoint['best_mbe']}")
    print(" *********************************************************************")

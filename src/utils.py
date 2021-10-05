import pandas as pd
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score
from src.metrics import mbe


def get_cat_emb_dims(cat_data: pd.DataFrame):
    cols = cat_data.columns
    col_to_emb = [
        n for n in cols if len(cat_data.loc[:, n].unique()) > 2 and n != "install" and n != "id"
    ]
    cat_dims = [len(cat_data.loc[:, col].unique()) for col in col_to_emb]
    emb_dims = [(x + 2, min(25, (x + 1) // 2)) for x in cat_dims]
    return emb_dims, col_to_emb


def init_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_performance(y_true, probs, losses):
    roc_auc = roc_auc_score(y_true=y_true, y_score=probs)
    loss = np.mean(losses)
    mbe_score = mbe(y_true=y_true, prob=probs)
    out = {
        "loss": loss,
        "roc_auc": roc_auc,
        "mbe": mbe_score,
    }
    return out

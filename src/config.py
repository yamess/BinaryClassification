import torch

config = {
    "SEED": 64,
    "TRAIN_DATA_PATH": "data/train_data.csv",
    "CHECKPOINT_PATH": "models/checkpoint.pt",
    "LOGS_PATH": "logs/metrics_logs.csv",
    "MODEL_PATH": "models/model.pt",
    "N_EPOCHS": 50,
    "TRAIN_BS": 512,  # 16,
    "VALID_BS": 5000,
    "CUT_POINT": 0.5,
    "LR": 0.005,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

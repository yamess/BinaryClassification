import torch

from src.utils import model_performance


def train(model, dataloader, optimizer, criterion, device):
    y_trues = []
    losses = []
    probs = []
    model.train()
    for batch in dataloader:
        x_emb = batch["x_emb"].to(device)
        x_cont = batch["x_cont"].to(device)
        y_true = batch["y"].to(device)

        optimizer.zero_grad()
        logits = model(x_cont, x_emb)
        logits = logits.squeeze(1)

        loss = criterion(logits, y_true.float())
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().item())
        probs.extend(torch.sigmoid(logits).detach().tolist())
        y_trues.extend(y_true.detach().cpu().tolist())
    train_performance = model_performance(
        y_true=y_trues,
        probs=probs,
        losses=losses,
    )
    return train_performance


def validate(model, dataloader, device, criterion):
    losses = []
    y_trues = []
    probs = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x_emb = batch["x_emb"].to(device)
            x_cont = batch["x_cont"].to(device)
            y_true = batch["y"].to(device)

            logits = model(x_cont, x_emb)

            logits = logits.squeeze(1)
            loss = criterion(logits, y_true.float())
            losses.append(loss.detach().cpu().item())
            probs.extend(torch.sigmoid(logits).detach().cpu().tolist())
            y_trues.extend(y_true.detach().cpu().tolist())
        valid_performance = model_performance(
            y_true=y_trues,
            probs=probs,
            losses=losses
        )
        return valid_performance, probs


def predict_proba(model, dataloader, device):
    probs = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x_emb = batch["x_emb"].to(device)
            x_cont = batch["x_cont"].to(device)
            logits = model(x_cont, x_emb)
            probs.extend(torch.sigmoid(logits).detach().cpu().tolist())
    return probs

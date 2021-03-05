import torch
import torch.nn as nn
from tqdm import tqdm 

def train(data_loader, model, optimizer, device):
    """
    This function does training for one epoch
    :param data_loader: the pytorch dataloader
    :param model: pytorch model
    :param optimizer: optimizer like adam, sgd, ...
    :param device: gpu/cpu
    """
    model.train()
    for data in data_loader:
        inputs = data["image"]
        targets = data["targets"]
        # Move data into device
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.BCEWithLogitsLoss()(outputs, targets.view(-1,1))
        # loss = nn.BCELoss()(outputs, targets.view(-1,1)) # Try different loss function
        loss.backward()
        optimizer.step()


def evaluate(data_loader, model, device):
    """
    This function does training for one epoch
    :param data_loader: the pytorch dataloader
    :param model: pytorch model
    :param device: gpu/cpu
    """
    model.eval()

    final_targets = []
    final_outputs = []

    with torch.no_grad():
        for data in data_loader:
            inputs = data["image"]
            targets = data["targets"]
            # Move data into device
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(inputs)
            targets = targets.detach().cpu().numpy().tolist()

            final_targets.extend(targets)
            final_outputs.extend(outputs)

    return final_outputs, final_targets


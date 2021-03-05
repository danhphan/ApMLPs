import torch
import torch.nn as nn

def loss_fn(outputs, targets):
    """
    This function returns the loss.
    """
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    for data in data_loader:
        ids = data["ids"]
        token_type_ids = data["token_type_ids"]
        mask = data["mask"]
        targets = data["targets"]
        # move to device
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        ouputs = model(ids, mask, token_type_ids)
        loss = loss_fn(outputs, targets)
        loss.backward()
        
        optimizer.step()
        scheduler.step()

def eval_fn(data_loader, model, device):
    model.eval()
    final_targets = []
    final_outputs = []
    with torch.no_grad():
        for data in data_loader:
            ids = data["ids"]
            token_type_ids = data["token_type_ids"]
            mask = data["mask"]
            targets = data["targets"]
            # move to device
            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)
            targets = targets.cpu().detach()
            final_targets.extend(targets.numpy().tolist())
            outputs = torch.sigmoid(outputs).cpu().detach()
            final_outputs.extend(outputs.numpy().tolist())
    return final_outputs, final_targets

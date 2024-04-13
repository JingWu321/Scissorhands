import torch
import gc
import numpy as np
from sklearn import linear_model, model_selection


## MIA
def compute_losses(net, loader, DEVICE):
    """Auxiliary function to compute per-sample losses"""

    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        logits = net(inputs)
        losses = criterion(logits, targets).detach().cpu().numpy()
        for l in losses:
            all_losses.append(l)
        torch.cuda.empty_cache()
        gc.collect()

    return np.array(all_losses)

def simple_mia(sample_loss, members, n_splits=10, random_state=0):

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )

def get_mia(net, member_loader, nonmember_loader, DEVICE, n_splits=10, random_state=0):
    loss_mem = compute_losses(net, member_loader, DEVICE)
    loss_non = compute_losses(net, nonmember_loader, DEVICE)

    # make sure we have a balanced dataset for the MIA
    if len(loss_mem) > len(loss_non):
        np.random.shuffle(loss_mem)
        loss_mem = loss_mem[: len(loss_non)]
    else:
        np.random.shuffle(loss_non)
        loss_non = loss_non[: len(loss_mem)]

    samples_loss = np.concatenate((loss_mem, loss_non)).reshape((-1, 1))
    label_members = [1] * len(loss_mem) + [0] * len(loss_non)
    mia_scores = simple_mia(samples_loss, label_members, n_splits, random_state)
    return mia_scores.mean()


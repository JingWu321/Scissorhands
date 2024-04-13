from trainer import train

from .impl import iterative_unlearn
import torch


@iterative_unlearn
def retrain(data_loaders, model, criterion, optimizer, epoch, args, mask):
    retain_loader = data_loaders["retain"]
    return train(retain_loader, model, criterion, optimizer, epoch, args, mask)


@iterative_unlearn
def raw(data_loaders, model, criterion, optimizer, epoch, args, mask):
    retain_loader = data_loaders["retain"]
    forget_loader = data_loaders["forget"]
    full_set = retain_loader.dataset + forget_loader.dataset
    full_loader = torch.utils.data.DataLoader(
        full_set, batch_size=retain_loader.batch_size, shuffle=True
    )
    return train(full_loader, model, criterion, optimizer, epoch, args, mask)



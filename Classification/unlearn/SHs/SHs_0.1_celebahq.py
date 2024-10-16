import time
import copy
from copy import deepcopy
import math
import gc

import numpy as np
import torch
import torch.nn as nn
import utils
from itertools import zip_longest
import quadprog

from .impl import iterative_unlearn


# projection
# https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py
def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().contiguous().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    # print("memories_np shape:{}".format(memories_np.shape))
    # print("gradient_np shape:{}".format(gradient_np.shape))
    t = memories_np.shape[0]  # task mums
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0] # get the optimal solution of v~
    x = np.dot(v, memories_np) + gradient_np  # g~ = v*GT +g
    # gradient.copy_(torch.Tensor(x).view(-1))
    new_grad = torch.Tensor(x).view(-1)
    return new_grad


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def re_init_weights(shape, device):
    mask = torch.empty(shape, requires_grad=False, device=device)
    if len(mask.shape) < 2:
        mask = torch.unsqueeze(mask, 1)
        # nn.init.xavier_uniform_(mask)
        nn.init.kaiming_uniform_(mask, a=math.sqrt(5))
        mask = torch.squeeze(mask, 1)
    else:
        # nn.init.xavier_uniform_(mask)
        nn.init.kaiming_uniform_(mask, a=math.sqrt(5))
    return mask


def create_dense_mask(net, device, value=1):
    for param in net.parameters():
        param.data[param.data == param.data] = value
    net.to(device)
    return net


def snip(net, dataloader, sparsity, prune_num, device):
    criterion = nn.CrossEntropyLoss()

    # compute grads
    grads = [torch.zeros_like(p) for p in net.parameters()]
    for ii in range(prune_num):
        inputs, targets = next(iter(dataloader))
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        net.zero_grad()
        loss.backward()

        j = 0
        for name, param in net.named_parameters():
            if (param.grad is not None):
                grads[j] += (param.grad.data).abs()
            j += 1
        torch.cuda.empty_cache()
        gc.collect()

    # compute saliences to get the threshold
    weights = [p for p in net.parameters()]
    mask_ = create_dense_mask(copy.deepcopy(net), device, value=1)
    with torch.no_grad():
        abs_saliences = [(grad * weight).abs() for weight, grad in zip(weights, grads)]
        saliences = [saliences.view(-1).cpu() for saliences in abs_saliences]
        saliences = torch.cat(saliences)
        # threshold = np.percentile(saliences, sparsity * 100) # kx100-th percentile
        threshold = float(saliences.kthvalue(int(sparsity * saliences.shape[0]))[0]) # k-th smallest value
        if (threshold >= saliences.max() - 1e-12) or (threshold <= saliences.min() + 1e-12):
            threshold = (saliences.max() - saliences.min()) / 2.

        # get mask to prune the weights
        for j, param in enumerate(mask_.parameters()):
            indx = (abs_saliences[j] > threshold) # prune for forget data
            param.data[indx] = 0

        # update the weights of the original network with the mask
        for (name, param), (m_param) in zip(net.named_parameters(), mask_.parameters()):
            # if ('bn' not in name) and (('layer3' in name) or ('layer4' in name) or ('fc' in name)):
            if ('bn' not in name) and (('layer4' in name) or ('fc' in name)):
                if ('weight' in name):
                    re_init_param = re_init_weights(param.data.shape, device)
                elif ('bias' in name):
                    re_init_param = torch.nn.init.zeros_(torch.empty(param.data.shape, device=device))
                param.data = param.data * m_param.data + re_init_param.data * (1 - m_param.data)

    return net



@iterative_unlearn
def SHs(data_loaders, model, criterion, optimizer, epoch, args, mask=None):

    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    device = torch.device(f"cuda:{int(args.gpu)}")
    # condition = lambda n: ('layer3' in n or 'layer4' in n or 'fc' in n)
    condition = lambda n: ('layer4' in n or 'fc' in n)

    # prune via snip
    if epoch == 0:
        model = snip(model, forget_loader, args.sparsity, args.prune_num, device=device)

    # get the gradient w.r.t the proxy_model
    if args.project:
        proxy_model = deepcopy(model).to(device)
        proxy_model.eval()
        g_o = []
        for ii in range(args.memory_num):
            image_f, target_f = next(iter(forget_loader))
            image_f, target_f = image_f.to(device), target_f.to(device)
            output_f = proxy_model(image_f)
            loss_f = -criterion(output_f, target_f)
            loss_f.backward()
            grad_o = []
            for n, param in proxy_model.named_parameters():
                if param.requires_grad and condition(n):
                    grad_o.append(param.grad.detach().view(-1))
            g_o.append(torch.cat(grad_o))
            torch.cuda.empty_cache()
            gc.collect()
        g_o = torch.stack(g_o, dim=1)
        # print(f'g_o: {g_o.shape}')

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top1_u = utils.AverageMeter()
    model.train()
    start = time.time()
    loader_len = max(len(forget_loader), len(retain_loader))
    if epoch < args.warmup:
        utils.warmup_lr(epoch, i+1, optimizer,
                        one_epoch_step=loader_len, args=args)

    i = 0
    for data_r, data_u in zip_longest(retain_loader, forget_loader, fillvalue=None):
        i += 1

        if data_r is None:
            break
        elif (data_u is None) and (data_r is not None):
            image, target = data_r
            image, target = image.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(image)
            # fms_r, output = model(image, is_feature=1)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            with torch.no_grad():
                output = output.float()
                loss = loss.float()
                prec = utils.accuracy(output.data, target)[0]
                losses.update(loss.item(), image.size(0))
                top1.update(prec.item(), image.size(0))
                torch.cuda.empty_cache()
                gc.collect()

        else:
            image_r, target_r = data_r
            image_u, target_u = data_u
            image_r, target_r = image_r.to(device), target_r.to(device)
            image_u, target_u = image_u.to(device), target_u.to(device)

            # compute output and loss
            optimizer.zero_grad()
            output_r = model(image_r)
            output_u = model(image_u)
            # fms_r, output_r = model(image_r, is_feature=1)
            # fms_u, output_u = model(image_u, is_feature=1)

            loss_r = criterion(output_r, target_r)
            loss_u = criterion(output_u, target_u)
            loss = loss_r - args.lam * loss_u

            loss.backward()

            # if args.project:
            if args.project and (i % 10 == 0):
                # get the gradient w.r.t the pruned model
                grad_f = []
                for n, param in model.named_parameters():
                    if param.requires_grad and condition(n):
                        grad_f.append(param.grad)
                g_f = torch.cat(list(map(lambda g: g.detach().view(-1), grad_f)))
                # print(f'g_f: {g_f.shape}')

                # compute the dot product of the gradients
                dotg = torch.mm(g_f.unsqueeze(0), g_o)
                # print(f'dotg: {(dotg < 0).sum()}')
                if args.project and ((dotg < 0).sum() != 0):
                    # project the gradient
                    grad_new = project2cone2(g_f.unsqueeze(0), g_o)
                    # overwrite the gradient
                    pointer = 0
                    for n, p in model.named_parameters():
                        if p.requires_grad and condition(n):
                            this_grad = grad_new[pointer:pointer + p.numel()].view(p.grad.data.size()).to(device)
                            p.grad.data.copy_(this_grad)
                            pointer += p.numel()

            # update the weights
            optimizer.step()
            # measure accuracy and record loss
            with torch.no_grad():
                output_r = output_r.float()
                output_u = output_u.float()
                loss = loss.float()
                prec_r = utils.accuracy(output_r.data, target_r)[0]
                prec_u = utils.accuracy(output_u.data, target_u)[0]
                losses.update(loss.item(), image_r.size(0) + image_u.size(0))
                top1.update(prec_r.item(), image_r.size(0))
                top1_u.update(prec_u.item(), image_u.size(0))
                torch.cuda.empty_cache()
                gc.collect()

            if (i + 1) % args.print_freq == 0:
                print(f'prec_u: {top1_u.val:.3f} ({top1_u.avg:.3f}), loss_u: {args.lam * loss_u:.4f}, loss_r: {loss_r:.4f}')


        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Time {3:.2f}'.format(
                        epoch, i, loader_len, end-start, loss=losses, top1=top1))
            start = time.time()

    return top1.avg





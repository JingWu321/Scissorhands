import argparse
import os
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from train_scripts.convertModels import savemodelDiffusers
from train_scripts.dataset import (
    setup_nsfw_data,
    setup_model,
)
from ldm.models.diffusion.ddim import DDIMSampler
from tqdm import tqdm

import gc
from timm.utils import AverageMeter
from timm.models.layers import trunc_normal_

import quadprog
import copy
import timm
import math


word_wear = "a photo of a person wearing clothes"


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


def create_dense_mask(net, device, value=1):
    for param in net.parameters():
        param.data[param.data == param.data] = value
    net.to(device)
    return net


def snip(model, dataloader, sparsity, prune_num, device):
    criterion = torch.nn.MSELoss()

    # compute grads
    grads = [torch.zeros_like(p) for p in model.model.diffusion_model.parameters()]
    for ii in range(prune_num):
        forget_batch = next(iter(dataloader))
        loss = model.shared_step(forget_batch)[0]
        model.model.diffusion_model.zero_grad()
        loss.backward()

        with torch.no_grad():
            j = 0
            for n, param in  model.model.diffusion_model.named_parameters():
                if (param.grad is not None):
                    grads[j] += (param.grad.data).abs()
                j += 1
            torch.cuda.empty_cache()
            gc.collect()

    # compute saliences to get the threshold
    weights = [p for p in model.model.diffusion_model.parameters()]
    mask_ = create_dense_mask(copy.deepcopy(model.model.diffusion_model), device, value=1)
    with torch.no_grad():
        abs_saliences = [(grad * weight).abs() for weight, grad in zip(weights, grads)]
        saliences = [saliences.view(-1).cpu() for saliences in abs_saliences]
        saliences = torch.cat(saliences)
        threshold = float(saliences.kthvalue(int(sparsity * saliences.shape[0]))[0]) # k-th smallest value

        # get mask to prune the weights
        for j, param in enumerate(mask_.parameters()):
            indx = (abs_saliences[j] > threshold) # prune for forget data
            param.data[indx] = 0

        # update the weights of the original network with the mask
        for (n, param), (m_param) in zip(model.model.diffusion_model.named_parameters(), mask_.parameters()):
            if ("attn2" in n):
            # if (n.startswith("out.")) or ("attn2" in n) or ("time_embed" in n):
            #     pass
            # else:
                # print('snip', n, m_param.data.sum())
                mask = torch.empty(param.data.shape, device=device)
                if ('weight' in n):
                    # re_init_param = torch.nn.init.kaiming_uniform_(mask, a=math.sqrt(5))
                    re_init_param = trunc_normal_(mask, std=.02)
                elif ('bias' in n):
                    re_init_param = torch.nn.init.zeros_(mask)
                param.data = param.data * m_param.data + re_init_param.data * (1 - m_param.data)

    return model


def SHs(classes,
        train_method,
        batch_size,
        epochs,
        sparsity,
        lam,
        project,
        memory_num,
        prune_num,
        lr,
        config_path,
        ckpt_path,
        mask_path,
        diffusers_config_path,
        device,
        image_size,
        ddim_steps,
):
    # MODEL TRAINING SETUP
    model = setup_model(config_path, ckpt_path, device)
    criteria = torch.nn.MSELoss()
    forget_dl, remain_dl = setup_nsfw_data(
        batch_size, forget_path='./dataFolder/NSFW',
        remain_path='./dataFolder/NotNSFW', image_size=image_size)

    # choose parameters to train based on train_method
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        if train_method == "noxattn":
            if name.startswith("out.") or "attn2" in name or "time_embed" in name:
                pass
            else:
                print(name)
                parameters.append(param)
        # train only self attention layers
        if train_method == "selfattn":
            if "attn1" in name:
                print(name)
                parameters.append(param)
        # train only x attention layers
        if train_method == "xattn":
            if "attn2" in name:
                print(name)
                parameters.append(param)
        # train all layers
        if train_method == "full":
            print(name)
            parameters.append(param)
        # train all layers except time embed layers
        if train_method == "notime":
            if not (name.startswith("out.") or "time_embed" in name):
                print(name)
                parameters.append(param)
        if train_method == "xlayer":
            if "attn2" in name:
                if "output_blocks.6." in name or "output_blocks.8." in name:
                    print(name)
                    parameters.append(param)
        if train_method == "selflayer":
            if "attn1" in name:
                if "input_blocks.4." in name or "input_blocks.7." in name:
                    print(name)
                    parameters.append(param)

    # prune via snip
    model = snip(model, forget_dl, sparsity, prune_num, device)
    # get the gradient w.r.t the proxy_model
    # condition = lambda n: ('attn2' in n) and ('output_blocks.11' in n) and ('output_blocks.10' in n) and ('output_blocks.9' in n)
    condition = lambda n: ('attn2' in n)
    # condition = lambda n: (n.startswith("out.") or "attn2" in n or "time_embed" in n)
    if project:
        proxy_model = copy.deepcopy(model).to(device)
        proxy_model.eval()
        g_o = []
        for ii in range(memory_num):
            forget_batch = next(iter(forget_dl))
            loss = -proxy_model.shared_step(forget_batch)[0]
            loss.backward()
            grad_o = []
            for n, param in proxy_model.model.diffusion_model.named_parameters():
                if param.grad is not None:
                    if condition(n):
                    #     pass
                    # else:
                        grad_o.append(param.grad.detach().view(-1))
            g_o.append(torch.cat(grad_o))
            torch.cuda.empty_cache()
            gc.collect()
        g_o = torch.stack(g_o, dim=1)
        # print(f'g_o: {g_o.shape}')

    # set model to train
    model.train()
    losses = []
    losses_e = AverageMeter()
    optimizer = torch.optim.Adam(parameters, lr=lr)

    if mask_path:
        mask = torch.load(mask_path)
        name = f"compvis-nsfw-SHs-mask-method_{train_method}-lr_{lr}_S{sparsity}_P{prune_num}_M{memory_num}_lam_{lam}_E{epochs}"
    else:
        name = f"compvis-nsfw-SHs-method_{train_method}-lr_{lr}_S{sparsity}_P{prune_num}_M{memory_num}_lam_{lam}_E{epochs}"

    # NSFW Removal
    word_wear = "a photo of a person wearing clothes"
    word_print = 'nsfw'.replace(" ", "")

    # TRAINING CODE
    step = 0
    for epoch in range(epochs):
        with tqdm(total=len(forget_dl)) as time:
            for i, batch in enumerate(forget_dl):
                model.train()
                optimizer.zero_grad()
                forget_batch = next(iter(forget_dl))
                remain_batch = next(iter(remain_dl))

                loss_r = model.shared_step(remain_batch)[0]

                forget_input, forget_emb = model.get_input(
                    forget_batch, model.first_stage_key
                )
                pseudo_prompts = [word_wear] * forget_batch['jpg'].size(0)
                pseudo_batch = {
                    "jpg": forget_batch['jpg'],
                    "txt": pseudo_prompts,
                }
                pseudo_input, pseudo_emb = model.get_input(
                    pseudo_batch, model.first_stage_key
                )

                t = torch.randint(0, model.num_timesteps, (forget_input.shape[0],), device=model.device,).long()
                noise = torch.randn_like(forget_input, device=model.device)
                forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)
                forget_out = model.apply_model(forget_noisy, t, forget_emb)
                pseudo_noisy = model.q_sample(x_start=pseudo_input, t=t, noise=noise)
                pseudo_out = model.apply_model(pseudo_noisy, t, pseudo_emb).detach()

                loss_u = criteria(forget_out, pseudo_out)
                loss = loss_r + lam * loss_u
                loss.backward()
                losses.append(loss.item() / batch_size)
                losses_e.update(loss.item())

                if project and (i % 10 == 0):
                    # get the gradient w.r.t the pruned model
                    grad_f = []
                    for n, param in model.model.diffusion_model.named_parameters():
                        # if (param.grad is not None) and condition(n):
                        if param.grad is not None:
                            if condition(n):
                            #     pass
                            # else:
                                grad_f.append(param.grad)
                    g_f = torch.cat(list(map(lambda g: g.detach().view(-1), grad_f)))
                    # print(f'g_f: {g_f.shape}')

                    # compute the dot product of the gradients
                    dotg = torch.mm(g_f.unsqueeze(0), g_o)
                    # print(f'dotg: {(dotg < 0).sum()}')
                    if ((dotg < 0).sum() != 0):
                        grad_new = project2cone2(g_f.unsqueeze(0), g_o)
                        # overwrite the gradient
                        pointer = 0
                        for n, p in model.model.diffusion_model.named_parameters():
                            # if (p.grad is not None) and condition(n):
                            if param.grad is not None:
                                if condition(n):
                                #     pass
                                # else:
                                    this_grad = grad_new[pointer:pointer + p.numel()].view(p.grad.data.size()).to(device)
                                    p.grad.data.copy_(this_grad)
                                    pointer += p.numel()

                optimizer.step()
                step += 1
                torch.cuda.empty_cache()
                gc.collect()

                if (step+1) % 55 == 0:
                #     print(f"step: {step}, loss: {loss:.4f}, loss_u: {loss_u:.4f}, loss_r: {loss_r:.4f}, losses_e: {losses_e.avg:.4f}")
                    save_history(losses, name, word_print)
                #     model.eval()
                #     save_model(model, name, step, save_compvis=True, save_diffusers=True, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)

                time.set_description("Epoch %i" % epoch)
                time.set_postfix(loss=loss.item() / batch_size)
                sleep(0.1)
                time.update(1)

        model.eval()
        save_model(model, name, epoch, save_compvis=True, save_diffusers=True, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)

    model.eval()
    save_model(
        model,
        name,
        None,
        save_compvis=True,
        save_diffusers=True,
        compvis_config_file=config_path,
        diffusers_config_file=diffusers_config_path,
    )
    save_history(losses, name, word_print)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_loss(losses, path, word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f"{word}_loss")
    plt.legend(loc="upper left")
    plt.title("Average loss in trainings", fontsize=20)
    plt.xlabel("Data point", fontsize=16)
    plt.ylabel("Loss value", fontsize=16)
    plt.savefig(path)


def save_model(
    model,
    name,
    num,
    compvis_config_file=None,
    diffusers_config_file=None,
    device="cpu",
    save_compvis=True,
    save_diffusers=True,
):
    # SAVE MODEL
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f"{folder_path}/{name}-epoch_{num}.pt"
    else:
        path = f"{folder_path}/{name}.pt"
    if save_compvis:
        torch.save(model.state_dict(), path)

    if save_diffusers:
        print("Saving Model in Diffusers Format")
        savemodelDiffusers(
            name, compvis_config_file, diffusers_config_file, device=device, num=num,
        )


def save_history(losses, name, word_print):
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/loss.txt", "w") as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses, f"{folder_path}/loss.png", word_print, n=3)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train", description="train a stable diffusion model from scratch"
    )
    parser.add_argument(
        "--class_to_forget",
        help="class corresponding to concept to erase",
        type=str,
        required=True,
        default="0",
    )
    parser.add_argument(
        "--train_method", help="method of training", type=str, required=True
    )
    parser.add_argument(
        "--batch_size",
        help="batch_size used to train",
        type=int,
        required=False,
        default=8,
    )
    parser.add_argument(
        "--epochs", help="epochs used to train", type=int, required=False, default=5
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--project",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--memory_num",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--prune_num",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--lr",
        help="learning rate used to train",
        type=float,
        required=False,
        default=1e-5,
    )
    parser.add_argument(
        "--ckpt_path",
        help="ckpt path for stable diffusion v1-4",
        type=str,
        required=False,
        default="models/ldm/sd-v1-4-full-ema.ckpt",
    )
    parser.add_argument(
        "--mask_path",
        help="mask path for stable diffusion v1-4",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--config_path",
        help="config path for stable diffusion v1-4 inference",
        type=str,
        required=False,
        default="configs/stable-diffusion/v1-inference.yaml",
    )
    parser.add_argument(
        "--diffusers_config_path",
        help="diffusers unet config json path",
        type=str,
        required=False,
        default="diffusers_unet_config.json",
    )
    parser.add_argument(
        "--device",
        help="cuda devices to train on",
        type=str,
        required=False,
        default="4",
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--ddim_steps",
        help="ddim steps of inference used to train",
        type=int,
        required=False,
        default=50,
    )
    args = parser.parse_args()

    classes = int(args.class_to_forget)
    print(classes)
    train_method = args.train_method
    batch_size = args.batch_size
    epochs = args.epochs
    sparsity = args.sparsity
    lam = args.lam
    project = args.project
    memory_num = args.memory_num
    prune_num = args.prune_num
    lr = args.lr
    ckpt_path = args.ckpt_path
    mask_path = args.mask_path
    config_path = args.config_path
    diffusers_config_path = args.diffusers_config_path
    device = f"cuda:{int(args.device)}"
    image_size = args.image_size
    ddim_steps = args.ddim_steps

    SHs(
        classes,
        train_method,
        batch_size,
        epochs,
        sparsity,
        lam,
        project,
        memory_num,
        prune_num,
        lr,
        config_path,
        ckpt_path,
        mask_path,
        diffusers_config_path,
        device,
        image_size,
        ddim_steps,
    )



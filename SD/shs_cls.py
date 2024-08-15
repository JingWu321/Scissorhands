import argparse
import os
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from train_scripts.convertModels import savemodelDiffusers
from train_scripts.dataset import (
    setup_model,
    setup_remain_data,
    setup_forget_data,
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
import time

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


def snip(model, dataloader, sparsity, prune_num, device, descriptions):
    criterion = torch.nn.MSELoss()

    # compute grads
    model.eval()
    grads = [torch.zeros_like(p) for p in model.model.diffusion_model.parameters()]
    for ii in range(prune_num):
        forget_images, forget_labels = next(iter(dataloader))
        forget_prompts = [descriptions[label] for label in forget_labels]
        forget_batch = {"jpg": forget_images.permute(0, 2, 3, 1), "txt": forget_prompts}

        null_prompts = ["" for label in forget_labels]
        null_batch = {"jpg": forget_images.permute(0, 2, 3, 1), "txt": null_prompts}
        forget_input, forget_emb = model.get_input(
            forget_batch, model.first_stage_key
        )
        null_input, null_emb = model.get_input(null_batch, model.first_stage_key)
        t = torch.randint(
            0, model.num_timesteps, (forget_input.shape[0],), device=device
        ).long()
        noise = torch.randn_like(forget_input, device=device)
        forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)
        forget_out = model.apply_model(forget_noisy, t, forget_emb)
        null_out = model.apply_model(forget_noisy, t, null_emb)
        preds = (1 + 7.5) * forget_out - 7.5 * null_out
        loss = criterion(noise, preds)

        # loss = model.shared_step(forget_batch)[0]
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
            if ("attn1" in n) and ("output_blocks.4." in n or "output_blocks.7." in n):
                print((m_param.data == 0).sum())

                mask = torch.empty(param.data.shape, device=device)
                if ('weight' in n):
                    re_init_param = torch.nn.init.kaiming_uniform_(mask, a=math.sqrt(5))
                    # re_init_param = trunc_normal_(mask, std=.02)
                elif ('bias' in n):
                    re_init_param = torch.nn.init.zeros_(mask)
                param.data = param.data * m_param.data + re_init_param.data * (1 - m_param.data)

    return model


def SHs(class_to_forget,
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
    remain_dl, descriptions = setup_remain_data(class_to_forget, batch_size, image_size)
    forget_dl, _ = setup_forget_data(class_to_forget, batch_size, image_size)

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
            # print(name)
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
    model = snip(model, forget_dl, sparsity, prune_num, device, descriptions)

    # set model to train
    model.train()
    losses = []
    optimizer = torch.optim.Adam(parameters, lr=lr)

    if mask_path:
        mask = torch.load(mask_path)
        name = f"compvis-SHs-mask-class_{str(class_to_forget)}-method_{train_method}-lr_{lr}_S{sparsity}_P{prune_num}_M{memory_num}_lam_{lam}_E{epochs}"
    else:
        name = f"compvis-SHs-class_{str(class_to_forget)}-method_{train_method}-lr_{lr}_S{sparsity}_P{prune_num}_M{memory_num}_lam_{lam}_E{epochs}"

    # TRAINING CODE
    step = 0
    for epoch in range(epochs):
        with tqdm(total=len(forget_dl)) as time_1:
            for i, batch in enumerate(forget_dl):
                model.train()
                optimizer.zero_grad()

                forget_images, forget_labels = next(iter(forget_dl))
                remain_images, remain_labels = next(iter(remain_dl))
                remain_prompts = [descriptions[label] for label in remain_labels]
                forget_prompts = [descriptions[label] for label in forget_labels]

                remain_batch = {
                    "jpg": remain_images.permute(0, 2, 3, 1),
                    "txt": remain_prompts,
                }
                loss_r = model.shared_step(remain_batch)[0]

                forget_batch = {
                    "jpg": forget_images.permute(0, 2, 3, 1),
                    "txt": forget_prompts,
                }
                loss_u = model.shared_step(forget_batch)[0]

                loss = loss_r - lam * loss_u
                loss.backward()
                losses.append(loss.item() / batch_size)

                optimizer.step()
                step += 1
                torch.cuda.empty_cache()
                gc.collect()

                # if ((step+1) > 50) and ((step+1) % 10 == 0):
                if (step+1) % 20 == 0:
                    print(f"step: {step}, loss: {loss:.4f}, loss_r: {loss_r:.4f}, lam*loss_u: {lam * loss_u:.4f}")
                    save_history(losses, name, class_to_forget)
                if step == 79:
                    model.eval()
                    save_model(model, name, step, save_compvis=True, save_diffusers=True, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)

                time_1.set_description("Epoch %i" % epoch)
                time_1.set_postfix(loss=loss.item() / batch_size)
                sleep(0.1)
                time_1.update(1)

        # model.eval()
        # save_model(model, name, epoch, save_compvis=True, save_diffusers=True, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)

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
    save_history(losses, name, class_to_forget)


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
        path = f"{folder_path}/{name}-step_{num}.pt"
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



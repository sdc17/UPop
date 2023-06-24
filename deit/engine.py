# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from .losses import DistillationLoss
import utils

from torch import nn

import io, os


def print_compression_statistics(model):
    mask_attn_list, mask_mlp_list = [], []
    reserved_ratio = lambda x: (torch.count_nonzero(x) / torch.numel(x)).item()
    for i in range(model.module.layers):
        mask_attn_list.append(getattr(model.module.blocks, str(i)).attn.alpha.data.view(-1))
        mask_mlp_list.append(getattr(model.module.blocks, str(i)).mlp.alpha.data.view(-1))
    print_format = lambda x: [round(i * 100, 2) for i in x]
    print('mask_attn_list:  ', print_format([reserved_ratio(x) for x in mask_attn_list]))
    print('mask_mlp_list:  ', print_format([reserved_ratio(x) for x in mask_mlp_list]))
    print('mask_attn: ', reserved_ratio(torch.cat(mask_attn_list)))
    print('mask_mlp: ', reserved_ratio(torch.cat(mask_mlp_list)))
    

def get_sparsity_loss(model, layers):
    sparsity_loss_attn, sparsity_loss_mlp = 0, 0
    for i in range(layers):
        sparsity_loss_attn += torch.sum(torch.abs(getattr(model.module.blocks, str(i)).attn.alpha))
        sparsity_loss_mlp += torch.sum(torch.abs(getattr(model.module.blocks, str(i)).mlp.alpha))
    return sparsity_loss_attn, sparsity_loss_mlp


def compress(model, search_model):
    for i in range(search_model.layers):
        # mlp
        in_features = getattr(model.blocks, str(i)).mlp.fc1.weight.shape[-1]
        out_features = getattr(model.blocks, str(i)).mlp.fc2.weight.shape[0]
        alpha = torch.squeeze(getattr(search_model.blocks, str(i)).mlp.alpha.data)
        hidden_features = torch.count_nonzero(alpha)
        getattr(model.blocks, str(i)).mlp.fc1 = nn.Linear(in_features, hidden_features)
        getattr(model.blocks, str(i)).mlp.fc1.weight.data = getattr(search_model.blocks, str(i)).mlp.fc1.weight.data[alpha==1,:]
        getattr(model.blocks, str(i)).mlp.fc1.bias.data = getattr(search_model.blocks, str(i)).mlp.fc1.bias.data[alpha==1]
        getattr(model.blocks, str(i)).mlp.fc2 = nn.Linear(hidden_features, out_features)
        getattr(model.blocks, str(i)).mlp.fc2.weight.data = getattr(search_model.blocks, str(i)).mlp.fc2.weight.data[:, alpha==1]
        getattr(model.blocks, str(i)).mlp.fc2.bias.data = getattr(search_model.blocks, str(i)).mlp.fc2.bias.data

        # attn
        in_features = getattr(model.blocks, str(i)).attn.qkv.weight.shape[-1]
        out_features = getattr(model.blocks, str(i)).attn.proj.weight.shape[0]
        alpha = torch.squeeze(getattr(search_model.blocks, str(i)).attn.alpha.data)
        hidden_features = torch.count_nonzero(alpha)
        parameter_ratio = 3*getattr(model.blocks, str(i)).attn.num_heads
        getattr(model.blocks, str(i)).attn.qkv = nn.Linear(in_features, hidden_features*parameter_ratio)
        getattr(model.blocks, str(i)).attn.qkv.weight.data = \
            getattr(search_model.blocks, str(i)).attn.qkv.weight.data[alpha.repeat(parameter_ratio)==1,:]
        getattr(model.blocks, str(i)).attn.qkv.weight.bias = \
            getattr(search_model.blocks, str(i)).attn.qkv.bias.data[alpha.repeat(parameter_ratio)==1]
        getattr(model.blocks, str(i)).attn.proj = nn.Linear(hidden_features, out_features)
        getattr(model.blocks, str(i)).attn.proj.weight.data = \
            getattr(search_model.blocks, str(i)).attn.proj.weight.data[:, alpha.repeat(parameter_ratio//3)==1]
        getattr(model.blocks, str(i)).attn.proj.bias.data = getattr(search_model.blocks, str(i)).attn.proj.bias.data


    
def update_alpha_parameters(model, layers, p, pi, print_info):
    alpha_grad_attn = torch.stack([getattr(model.module.blocks, str(i)).attn.alpha.grad for i in range(layers)])
    alpha_grad_mlp = torch.stack([getattr(model.module.blocks, str(i)).mlp.alpha.grad for i in range(layers)])

    standarlization = lambda x: (x - torch.mean(x)) / torch.std(x)
    alpha_grad_attn, alpha_grad_mlp = standarlization(alpha_grad_attn), standarlization(alpha_grad_mlp)
    alpha_grad = torch.cat([alpha_grad_attn.view(-1), alpha_grad_mlp.view(-1)])
    sorted_alpha_grad, indices = torch.sort(alpha_grad, descending=True)
    compression_weight = torch.ones_like(indices)
    compression_weight[indices < alpha_grad_attn.numel()] = 9234/769
    threshold = sorted_alpha_grad[torch.argmin(torch.abs(torch.cumsum(compression_weight, 0) - torch.sum(compression_weight)*pi))]
    
    def update(module, grad):
        mask = ((grad <= threshold) | (grad <= torch.min(grad)))
        module.data.copy_(mask + (~mask)*(1 - pi/p))

    for i in range(layers):
        update(getattr(model.module.blocks, str(i)).attn.alpha, alpha_grad_attn[i])
        update(getattr(model.module.blocks, str(i)).mlp.alpha, alpha_grad_mlp[i])

    if print_info:
        attn, mlp = [], []
        for i in range(layers):
            attn.append(getattr(model.module.blocks, str(i)).attn.alpha.flatten())
            mlp.append(getattr(model.module.blocks, str(i)).mlp.alpha.flatten())
        print('Current compression ratio of attn: ', 1-torch.mean(torch.cat(attn)))
        print('Current compression ratio of mlp: ', 1-torch.mean(torch.cat(mlp)))
        print('Current compression ratio: ', pi)  



def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None,
                    search=False, update_alpha=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if search:
        metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_sp_attn', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_sp_mlp', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        layers = model.module.layers

    header = 'Train Epoch: [{}]'.format(epoch) if not search else 'Search Epoch: [{}]'.format(epoch)
    print_freq = 10

    len_data_loader = len(data_loader)
    total_steps = len_data_loader*args.epochs if not search else len_data_loader*args.epochs_search

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
                    
        outputs = model(samples)
        loss = criterion(samples, outputs, targets)
        if search:
            sparsity_loss_attn, sparsity_loss_mlp = get_sparsity_loss(model, layers)
            metric_logger.update(loss_ce=loss.item()) 
            metric_logger.update(loss_sp_attn=args.w_sp_attn * sparsity_loss_attn.item()) 
            metric_logger.update(loss_sp_mlp=args.w_sp_mlp * sparsity_loss_mlp.item()) 
            loss += args.w_sp_attn * sparsity_loss_attn + args.w_sp_mlp * sparsity_loss_mlp
            step = epoch*len_data_loader+i
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()    
        
        if search and ((step > 0) and (step % args.interval == 0 or step == total_steps - 1)) and update_alpha:
            pi = args.p*((1-math.cos(math.pi*(step+1)/total_steps))/2)**(1/2)
            update_alpha_parameters(model, layers, args.p, pi, True)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compress(model, search_model):
    for i in range(search_model.layers):
        # mlp
        in_features = getattr(model.blocks, str(i)).mlp.fc1.weight.shape[-1]
        out_features = getattr(model.blocks, str(i)).mlp.fc2.weight.shape[0]
        alpha = torch.squeeze(getattr(search_model.blocks, str(i)).mlp.alpha.data)
        hidden_features = torch.count_nonzero(alpha)
        getattr(model.blocks, str(i)).mlp.fc1 = nn.Linear(in_features, hidden_features)
        getattr(model.blocks, str(i)).mlp.fc1.weight.data = getattr(search_model.blocks, str(i)).mlp.fc1.weight.data[alpha==1,:]
        getattr(model.blocks, str(i)).mlp.fc1.bias.data = getattr(search_model.blocks, str(i)).mlp.fc1.bias.data[alpha==1]
        getattr(model.blocks, str(i)).mlp.fc2 = nn.Linear(hidden_features, out_features)
        getattr(model.blocks, str(i)).mlp.fc2.weight.data = getattr(search_model.blocks, str(i)).mlp.fc2.weight.data[:, alpha==1]
        getattr(model.blocks, str(i)).mlp.fc2.bias.data = getattr(search_model.blocks, str(i)).mlp.fc2.bias.data

        # attn
        in_features = getattr(model.blocks, str(i)).attn.qkv.weight.shape[-1]
        out_features = getattr(model.blocks, str(i)).attn.proj.weight.shape[0]
        alpha = torch.squeeze(getattr(search_model.blocks, str(i)).attn.alpha.data)
        hidden_features = torch.count_nonzero(alpha)
        parameter_ratio = 3*getattr(model.blocks, str(i)).attn.num_heads
        getattr(model.blocks, str(i)).attn.qkv = nn.Linear(in_features, hidden_features*parameter_ratio)
        getattr(model.blocks, str(i)).attn.qkv.weight.data = \
            getattr(search_model.blocks, str(i)).attn.qkv.weight.data[alpha.repeat(parameter_ratio)==1,:]
        getattr(model.blocks, str(i)).attn.qkv.weight.bias = \
            getattr(search_model.blocks, str(i)).attn.qkv.bias.data[alpha.repeat(parameter_ratio)==1]
        getattr(model.blocks, str(i)).attn.proj = nn.Linear(hidden_features, out_features)
        getattr(model.blocks, str(i)).attn.proj.weight.data = \
            getattr(search_model.blocks, str(i)).attn.proj.weight.data[:, alpha.repeat(parameter_ratio//3)==1]
        getattr(model.blocks, str(i)).attn.proj.bias.data = getattr(search_model.blocks, str(i)).attn.proj.bias.data



def prune_if_compressed(model, client, url_or_filename):
        
    if client is not None:
        with io.BytesIO(client.get(os.path.join('s3://BucketName/ProjectName', url_or_filename), enable_cache=True)) as f:
            checkpoint = torch.load(f, map_location='cpu')
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
    state_dict = checkpoint['model']

    for i in range(model.layers):
        # mlp
        if getattr(model.blocks, str(i)).mlp.fc1.weight.shape != state_dict['blocks.'+str(i)+'.mlp.fc1.weight'].shape:
            del getattr(model.blocks, str(i)).mlp.fc1
            getattr(model.blocks, str(i)).mlp.fc1 = nn.Linear(*state_dict['blocks.'+str(i)+'.mlp.fc1.weight'].shape[::-1])
            del getattr(model.blocks, str(i)).mlp.fc2
            getattr(model.blocks, str(i)).mlp.fc2 = nn.Linear(*state_dict['blocks.'+str(i)+'.mlp.fc2.weight'].shape[::-1])

        # attn
        if getattr(model.blocks, str(i)).attn.qkv.weight.shape != state_dict['blocks.'+str(i)+'.attn.qkv.weight'].shape:
            del getattr(model.blocks, str(i)).attn.qkv
            getattr(model.blocks, str(i)).attn.qkv = nn.Linear(*state_dict['blocks.'+str(i)+'.attn.qkv.weight'].shape[::-1])
            del getattr(model.blocks, str(i)).attn.proj
            getattr(model.blocks, str(i)).attn.proj = nn.Linear(*state_dict['blocks.'+str(i)+'.attn.proj.weight'].shape[::-1])

    torch.cuda.empty_cache()
    model.load_state_dict(state_dict, strict=False)
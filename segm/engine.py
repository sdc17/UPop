import torch
import math

from segm.utils.logger import MetricLogger
from segm.metrics import gather_data, compute_metrics
from segm.model import utils
from segm.data.utils import IGNORE_LABEL
import segm.utils.torch as ptu

from torch import nn

def print_compression_statistics(model):
    mask_attn_list, mask_mlp_list = [], []
    reserved_ratio = lambda x: (torch.count_nonzero(x) / torch.numel(x)).item()
    for i in range(model.module.encoder.n_layers):
        mask_attn_list.append(getattr(model.module.encoder.blocks, str(i)).attn.alpha.data.view(-1))
        mask_mlp_list.append(getattr(model.module.encoder.blocks, str(i)).mlp.alpha.data.view(-1))
    for i in range(model.module.decoder.n_layers):
        mask_attn_list.append(getattr(model.module.decoder.blocks, str(i)).attn.alpha.data.view(-1))
        mask_mlp_list.append(getattr(model.module.decoder.blocks, str(i)).mlp.alpha.data.view(-1))
    print_format = lambda x: [round(i * 100, 2) for i in x]
    print('mask_attn_list:  ', print_format([reserved_ratio(x) for x in mask_attn_list]))
    print('mask_mlp_list:  ', print_format([reserved_ratio(x) for x in mask_mlp_list]))
    print('mask_attn: ', reserved_ratio(torch.cat(mask_attn_list)))
    print('mask_mlp: ', reserved_ratio(torch.cat(mask_mlp_list)))
    

def get_sparsity_loss(model):
    sparsity_loss_attn, sparsity_loss_mlp = 0, 0
    for i in range(model.module.encoder.n_layers):
        sparsity_loss_attn += torch.sum(torch.abs(getattr(model.module.encoder.blocks, str(i)).attn.alpha))
        sparsity_loss_mlp += torch.sum(torch.abs(getattr(model.module.encoder.blocks, str(i)).mlp.alpha))
    for i in range(model.module.decoder.n_layers):
        sparsity_loss_attn += torch.sum(torch.abs(getattr(model.module.decoder.blocks, str(i)).attn.alpha))
        sparsity_loss_mlp += torch.sum(torch.abs(getattr(model.module.decoder.blocks, str(i)).mlp.alpha))
    return sparsity_loss_attn, sparsity_loss_mlp


def compress(model, search_model):
    for i in range(search_model.encoder.n_layers):
        # encoder mlp
        in_features = getattr(model.encoder.blocks, str(i)).mlp.fc1.weight.shape[-1]
        out_features = getattr(model.encoder.blocks, str(i)).mlp.fc2.weight.shape[0]
        alpha = torch.squeeze(getattr(search_model.encoder.blocks, str(i)).mlp.alpha.data)
        hidden_features = torch.count_nonzero(alpha)
        getattr(model.encoder.blocks, str(i)).mlp.fc1 = nn.Linear(in_features, hidden_features)
        getattr(model.encoder.blocks, str(i)).mlp.fc1.weight.data = getattr(search_model.encoder.blocks, str(i)).mlp.fc1.weight.data[alpha==1,:]
        getattr(model.encoder.blocks, str(i)).mlp.fc1.bias.data = getattr(search_model.encoder.blocks, str(i)).mlp.fc1.bias.data[alpha==1]
        getattr(model.encoder.blocks, str(i)).mlp.fc2 = nn.Linear(hidden_features, out_features)
        getattr(model.encoder.blocks, str(i)).mlp.fc2.weight.data = getattr(search_model.encoder.blocks, str(i)).mlp.fc2.weight.data[:, alpha==1]
        getattr(model.encoder.blocks, str(i)).mlp.fc2.bias.data = getattr(search_model.encoder.blocks, str(i)).mlp.fc2.bias.data

        # encoder attn
        in_features = getattr(model.encoder.blocks, str(i)).attn.qkv.weight.shape[-1]
        out_features = getattr(model.encoder.blocks, str(i)).attn.proj.weight.shape[0]
        alpha = torch.squeeze(getattr(search_model.encoder.blocks, str(i)).attn.alpha.data)
        hidden_features = torch.count_nonzero(alpha)
        parameter_ratio = 3*getattr(model.encoder.blocks, str(i)).attn.heads
        getattr(model.encoder.blocks, str(i)).attn.qkv = nn.Linear(in_features, hidden_features*parameter_ratio)
        getattr(model.encoder.blocks, str(i)).attn.qkv.weight.data = \
            getattr(search_model.encoder.blocks, str(i)).attn.qkv.weight.data[alpha.repeat(parameter_ratio)==1,:]
        getattr(model.encoder.blocks, str(i)).attn.qkv.bias.data = \
            getattr(search_model.encoder.blocks, str(i)).attn.qkv.bias.data[alpha.repeat(parameter_ratio)==1]
        getattr(model.encoder.blocks, str(i)).attn.proj = nn.Linear(hidden_features, out_features)
        getattr(model.encoder.blocks, str(i)).attn.proj.weight.data = \
            getattr(search_model.encoder.blocks, str(i)).attn.proj.weight.data[:, alpha.repeat(parameter_ratio//3)==1]
        getattr(model.encoder.blocks, str(i)).attn.proj.bias.data = getattr(search_model.encoder.blocks, str(i)).attn.proj.bias.data

    for i in range(search_model.decoder.n_layers):
        # decoder mlp
        in_features = getattr(model.decoder.blocks, str(i)).mlp.fc1.weight.shape[-1]
        out_features = getattr(model.decoder.blocks, str(i)).mlp.fc2.weight.shape[0]
        alpha = torch.squeeze(getattr(search_model.decoder.blocks, str(i)).mlp.alpha.data)
        hidden_features = torch.count_nonzero(alpha)
        getattr(model.decoder.blocks, str(i)).mlp.fc1 = nn.Linear(in_features, hidden_features)
        getattr(model.decoder.blocks, str(i)).mlp.fc1.weight.data = getattr(search_model.decoder.blocks, str(i)).mlp.fc1.weight.data[alpha==1,:]
        getattr(model.decoder.blocks, str(i)).mlp.fc1.bias.data = getattr(search_model.decoder.blocks, str(i)).mlp.fc1.bias.data[alpha==1]
        getattr(model.decoder.blocks, str(i)).mlp.fc2 = nn.Linear(hidden_features, out_features)
        getattr(model.decoder.blocks, str(i)).mlp.fc2.weight.data = getattr(search_model.decoder.blocks, str(i)).mlp.fc2.weight.data[:, alpha==1]
        getattr(model.decoder.blocks, str(i)).mlp.fc2.bias.data = getattr(search_model.decoder.blocks, str(i)).mlp.fc2.bias.data

        # decoder attn
        in_features = getattr(model.decoder.blocks, str(i)).attn.qkv.weight.shape[-1]
        out_features = getattr(model.decoder.blocks, str(i)).attn.proj.weight.shape[0]
        alpha = torch.squeeze(getattr(search_model.decoder.blocks, str(i)).attn.alpha.data)
        hidden_features = torch.count_nonzero(alpha)
        parameter_ratio = 3*getattr(model.decoder.blocks, str(i)).attn.heads
        getattr(model.decoder.blocks, str(i)).attn.qkv = nn.Linear(in_features, hidden_features*parameter_ratio)
        getattr(model.decoder.blocks, str(i)).attn.qkv.weight.data = \
            getattr(search_model.decoder.blocks, str(i)).attn.qkv.weight.data[alpha.repeat(parameter_ratio)==1,:]
        getattr(model.decoder.blocks, str(i)).attn.qkv.bias.data = \
            getattr(search_model.decoder.blocks, str(i)).attn.qkv.bias.data[alpha.repeat(parameter_ratio)==1]
        getattr(model.decoder.blocks, str(i)).attn.proj = nn.Linear(hidden_features, out_features)
        getattr(model.decoder.blocks, str(i)).attn.proj.weight.data = \
            getattr(search_model.decoder.blocks, str(i)).attn.proj.weight.data[:, alpha.repeat(parameter_ratio//3)==1]
        getattr(model.decoder.blocks, str(i)).attn.proj.bias.data = getattr(search_model.decoder.blocks, str(i)).attn.proj.bias.data

    
def update_alpha_parameters(model, p, pi, print_info):
    encoder_layers = model.module.encoder.n_layers
    encoder_alpha_grad_attn = torch.stack([getattr(model.module.encoder.blocks, str(i)).attn.alpha.grad for i in range(encoder_layers)])
    encoder_alpha_grad_mlp = torch.stack([getattr(model.module.encoder.blocks, str(i)).mlp.alpha.grad for i in range(encoder_layers)])
    decoder_layers = model.module.decoder.n_layers
    decoder_alpha_grad_attn = torch.stack([getattr(model.module.decoder.blocks, str(i)).attn.alpha.grad for i in range(decoder_layers)])
    decoder_alpha_grad_mlp = torch.stack([getattr(model.module.decoder.blocks, str(i)).mlp.alpha.grad for i in range(decoder_layers)])
    alpha_grad_attn = torch.cat([encoder_alpha_grad_attn, decoder_alpha_grad_attn], dim=0)
    alpha_grad_mlp = torch.cat([encoder_alpha_grad_mlp, decoder_alpha_grad_mlp], dim=0)

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

    for i in range(encoder_layers):
        update(getattr(model.module.encoder.blocks, str(i)).attn.alpha, alpha_grad_attn[i])
        update(getattr(model.module.encoder.blocks, str(i)).mlp.alpha, alpha_grad_mlp[i])
    for i in range(decoder_layers):
        update(getattr(model.module.decoder.blocks, str(i)).attn.alpha, alpha_grad_attn[encoder_layers+i])
        update(getattr(model.module.decoder.blocks, str(i)).mlp.alpha, alpha_grad_mlp[encoder_layers+i])

    if print_info:
        attn, mlp = [], []
        for i in range(encoder_layers):
            attn.append(getattr(model.module.encoder.blocks, str(i)).attn.alpha.flatten())
            mlp.append(getattr(model.module.encoder.blocks, str(i)).mlp.alpha.flatten())
        for i in range(decoder_layers):
            attn.append(getattr(model.module.decoder.blocks, str(i)).attn.alpha.flatten())
            mlp.append(getattr(model.module.decoder.blocks, str(i)).mlp.alpha.flatten())
        print('Current compression ratio of attn: ', 1-torch.mean(torch.cat(attn)))
        print('Current compression ratio of mlp: ', 1-torch.mean(torch.cat(mlp)))
        print('Current compression ratio: ', pi)  


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    lr_scheduler,
    epoch,
    amp_autocast,
    loss_scaler,
    args = None,
    search=False, 
    update_alpha=True
):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    logger = MetricLogger(delimiter="  ")
    header = 'Train Epoch: [{}]'.format(epoch) if not search else 'Search Epoch: [{}]'.format(epoch)
    print_freq = 100

    len_data_loader = len(data_loader)
    total_steps = len_data_loader*args['epochs'] if not search else len_data_loader*args['epochs_search']

    model.train()
    data_loader.set_epoch(epoch)
    num_updates = epoch * len(data_loader)
    for i, batch in enumerate(logger.log_every(data_loader, print_freq, header)):
        im = batch["im"].to(ptu.device)
        seg_gt = batch["segmentation"].long().to(ptu.device)

        seg_pred = model.forward(im)
        loss = criterion(seg_pred, seg_gt)
        if search:
            sparsity_loss_attn, sparsity_loss_mlp = get_sparsity_loss(model)
            logger.update(loss_ce=loss.item()) 
            logger.update(loss_sp_attn=args['w_sp_attn'] * sparsity_loss_attn.item()) 
            logger.update(loss_sp_mlp=args['w_sp_mlp'] * sparsity_loss_mlp.item()) 
            loss += args['w_sp_attn'] * sparsity_loss_attn + args['w_sp_mlp'] * sparsity_loss_mlp
            step = epoch*len_data_loader+i

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
            )
        else:
            loss.backward()
            optimizer.step()

        if search and ((step > 0) and (step % args['interval'] == 0 or step == total_steps - 1)) and update_alpha:
            pi = args['p']*((1-math.cos(math.pi*(step+1)/total_steps))/2)**(1/2)
            update_alpha_parameters(model, args['p'], pi, True)

        num_updates += 1
        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates)

        torch.cuda.synchronize()

        logger.update(
            loss=loss.item(),
            learning_rate=optimizer.param_groups[0]["lr"],
        )

    return logger


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    val_seg_gt,
    window_size,
    window_stride,
    amp_autocast,
):
    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module
    logger = MetricLogger(delimiter="  ")
    header = "Eval:"
    print_freq = 50

    val_seg_pred = {}
    model.eval()
    for batch in logger.log_every(data_loader, print_freq, header):
        ims = [im.to(ptu.device) for im in batch["im"]]
        ims_metas = batch["im_metas"]
        ori_shape = ims_metas[0]["ori_shape"]
        ori_shape = (ori_shape[0].item(), ori_shape[1].item())
        filename = batch["im_metas"][0]["ori_filename"][0]

        with amp_autocast():
            seg_pred = utils.inference(
                model_without_ddp,
                ims,
                ims_metas,
                ori_shape,
                window_size,
                window_stride,
                batch_size=1,
            )
            seg_pred = seg_pred.argmax(0)

        seg_pred = seg_pred.cpu().numpy()
        val_seg_pred[filename] = seg_pred

    val_seg_pred = gather_data(val_seg_pred)
    scores = compute_metrics(
        val_seg_pred,
        val_seg_gt,
        data_loader.unwrapped.n_cls,
        ignore_index=IGNORE_LABEL,
        distributed=ptu.distributed,
    )

    for k, v in scores.items():
        logger.update(**{f"{k}": v, "n": 1})

    return logger

import argparse
from importlib import import_module
import os
import ruamel_yaml as yaml
import numpy as np
import random
from pathlib import Path
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.blip_nlvr import blip_nlvr

import utils
from utils import cosine_lr_schedule, warmup_lr_schedule, print_params_and_flops
from data import create_dataset, create_sampler, create_loader

import io
# from petrel_client.client import Client
import math


def update_alpha_parameters(model, layers, p, pi, print_info=True):

    standarlization = lambda x: (x - torch.mean(x)) / torch.std(x)
    alpha_grad_attn = torch.stack([
        torch.cat([getattr(model.module.visual_encoder.blocks, str(i)).attn.alpha.grad for i in range(layers)]),
        torch.stack([getattr(model.module.text_encoder.encoder.layer, str(i)).attention.self.alpha.grad for i in range(layers)]),
        torch.stack([getattr(model.module.text_encoder.encoder.layer, str(i)).crossattention.self0.alpha.grad for i in range(layers)]),
        torch.stack([getattr(model.module.text_encoder.encoder.layer, str(i)).crossattention.self1.alpha.grad for i in range(layers)]),
    ])
    alpha_grad_mlp = torch.stack([
        torch.stack([getattr(model.module.visual_encoder.blocks, str(i)).mlp.alpha.grad for i in range(layers)]),
        torch.stack([getattr(model.module.text_encoder.encoder.layer, str(i)).intermediate.alpha.grad for i in range(layers)]),
    ])
    alpha_grad_attn, alpha_grad_mlp = standarlization(alpha_grad_attn), standarlization(alpha_grad_mlp)
    alpha_grad = torch.cat([alpha_grad_attn.view(-1), alpha_grad_mlp.view(-1)])
    sorted_alpha_grad, indices = torch.sort(alpha_grad, descending=True)
    compression_weight = torch.ones_like(indices)
    compression_weight[indices < alpha_grad_attn.numel()] = 36
    threshold = sorted_alpha_grad[torch.argmin(torch.abs(torch.cumsum(compression_weight, 0) - torch.sum(compression_weight)*pi))]
    
    def update(module, grad):
        mask = ((grad <= threshold) | (grad <= torch.min(grad)))
        module.data.copy_(mask + (~mask)*(1 - pi/p))

    for i in range(layers):
        update(getattr(model.module.visual_encoder.blocks, str(i)).attn.alpha, alpha_grad_attn[0, i].unsqueeze(0))
        update(getattr(model.module.text_encoder.encoder.layer, str(i)).attention.self.alpha, alpha_grad_attn[1, i])
        update(getattr(model.module.text_encoder.encoder.layer, str(i)).crossattention.self0.alpha, alpha_grad_attn[2, i])
        update(getattr(model.module.text_encoder.encoder.layer, str(i)).crossattention.self1.alpha, alpha_grad_attn[3, i])
        update(getattr(model.module.visual_encoder.blocks, str(i)).mlp.alpha, alpha_grad_mlp[0, i])
        update(getattr(model.module.text_encoder.encoder.layer, str(i)).intermediate.alpha, alpha_grad_mlp[1, i])

    if print_info:
        attn, mlp = [], []
        for i in range(layers):
            attn.append(getattr(model.module.visual_encoder.blocks, str(i)).attn.alpha.flatten())
            attn.append(getattr(model.module.text_encoder.encoder.layer, str(i)).attention.self.alpha.flatten())
            attn.append(getattr(model.module.text_encoder.encoder.layer, str(i)).crossattention.self0.alpha.flatten())
            attn.append(getattr(model.module.text_encoder.encoder.layer, str(i)).crossattention.self1.alpha.flatten())
            mlp.append(getattr(model.module.visual_encoder.blocks, str(i)).mlp.alpha.flatten())
            mlp.append(getattr(model.module.text_encoder.encoder.layer, str(i)).intermediate.alpha.flatten())
        print('Current compression ratio of attn: ', 1-torch.mean(torch.cat(attn)))
        print('Current compression ratio of mlp: ', 1-torch.mean(torch.cat(mlp)))
        print('Current compression ratio: ', pi)  


def train(model, data_loader, optimizer, epoch, device, config, search=False):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.7f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    if search:
        metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_sp_attn', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_sp_mlp', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch) if not search else 'Search Epoch: [{}]'.format(epoch)
    print_freq = 50   
    len_data_loader = len(data_loader)
    total_steps = len_data_loader*config['max_epoch']
    for i,(image0, image1, text, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        images = torch.cat([image0, image1], dim=0)
        images, targets = images.to(device), targets.to(device)   
        
        loss = model(images, text, targets=targets, train=True)
        if search:
            sparsity_loss_attn, sparsity_loss_mlp = model.module.get_sparsity_loss()
            metric_logger.update(loss_ce=loss.item()) 
            metric_logger.update(loss_sp_attn=config['w_sp_attn'] * sparsity_loss_attn.item()) 
            metric_logger.update(loss_sp_mlp=config['w_sp_mlp'] * sparsity_loss_mlp.item()) 
            loss += config['w_sp_attn'] * sparsity_loss_attn + config['w_sp_mlp'] * sparsity_loss_mlp

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    

        step = epoch*len_data_loader+i
        if search and (step % 50 == 0 or step == total_steps - 1):
            pi = config['p']*((1-math.cos(math.pi*(step+1)/total_steps))/2)**(1/2)
            update_alpha_parameters(model, 12 if config['vit']=='base' else 24, config['p'], pi)

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())  

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    for image0, image1, text, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = torch.cat([image0, image1], dim=0)
        images, targets = images.to(device), targets.to(device)   
        
        prediction = model(images, text, targets=targets, train=False)  
 
        _, pred_class = prediction.max(1)
        accuracy = (targets==pred_class).sum() / targets.size(0)
        
        metric_logger.meters['acc'].update(accuracy.item(), n=image0.size(0))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger.global_avg())   
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def main(args, config, client):
    utils.init_distributed_mode(args)    

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    config['pretrained'] = args.pretrained
    config['w_sp_attn'] = args.w_sp_attn / args.world_size
    config['w_sp_mlp'] = args.w_sp_mlp  /args.world_size
    config['max_epoch'] = args.epoch
    config['p'] = args.p
    if not args.evaluate:
        print('Target compression ratio: {}%'.format(config['p']*100))
        

    #### Dataset #### 
    print("Creating dataset")
    datasets = create_dataset('nlvr', config, client) 
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True,False,False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]
    
    batch_size=[config['batch_size_train'],config['batch_size_test'],config['batch_size_test']]
    train_loader, val_loader, test_loader = create_loader(datasets,samplers,batch_size=batch_size,
                                                          num_workers=[4,4,4],is_trains=[True,False,False], 
                                                          collate_fns=[None,None,None])
    
    if not args.evaluate:
        print("Creating model for searching")
        search_model = blip_nlvr(client=client, pretrained=config['pretrained'], image_size=config['image_size'], 
                                vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                                search = True)
        search_model = search_model.to(device)  
        print_params_and_flops('nlvr', search_model, device)
        search_model_without_ddp = search_model
        if args.distributed:
            search_model = torch.nn.parallel.DistributedDataParallel(search_model, device_ids=[args.gpu])
            search_model_without_ddp = search_model.module    
        optimizer = torch.optim.AdamW(
                params=[{'params':[param for name, param in list(search_model.named_parameters()) if not ('alpha' in name)]}], 
                lr=config['init_lr'], 
                weight_decay=config['weight_decay']
                )
        
        print("Start searching")
        for epoch in range(0, config['max_epoch']):
            if args.evaluate:
                break
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train(search_model, train_loader, optimizer, epoch, device, config, search=True) 
        dist.barrier()   
        search_model.module.print_compression_statistics()
    
        print("Creating model for training")
        model = blip_nlvr(client=client, pretrained='', image_size=config['image_size'], 
                            vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
        msg = model.load_state_dict(search_model_without_ddp.state_dict(), strict=False)
        model.compress(search_model_without_ddp)
    else:
        print("Creating model for evaluation")
        model = blip_nlvr(client=client, pretrained='', image_size=config['image_size'], 
                            vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                            evaluate=True)
        model.prune_if_compressed(client, config['pretrained'])

    model = model.to(device)   
    print_params_and_flops('nlvr', model, device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    
    if not args.evaluate:
        with torch.no_grad():
            val_stats = evaluate(model, val_loader, device, config)
            test_stats = evaluate(model, test_loader, device, config) 
            if utils.is_main_process(): 
                log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            }
                print("LOG: ", log_stats)

    print("Start training")
    best = 0
    best_epoch = 0
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            train_stats = train(model, train_loader, optimizer, epoch,  device, config) 
            
        val_stats = evaluate(model, val_loader, device, config)
        test_stats = evaluate(model, test_loader, device, config)  
        
        if utils.is_main_process():  
            if args.evaluate:                
                log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            }
                print("LOG: ", log_stats)
            else:       
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'val_{k}': v for k, v in val_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            }
                if float(val_stats['acc'])>best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        # 'optimizer': optimizer.state_dict(),
                        # 'config': config,
                        # 'epoch': epoch,
                    }
                    if client is not None:
                        with io.BytesIO() as f:
                            torch.save(save_obj, f)
                            client.put(os.path.join('s3://BucketName/ProjectName', args.output_dir, 'checkpoint_best.pth'), f.getvalue())
                    else:
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth')) 

                    best = float(val_stats['acc'])
                    best_epoch = epoch
                print("LOG: ", log_stats)
        if args.evaluate:             
            break            
        dist.barrier()   


    if utils.is_main_process():   
        print("LOG: best epoch: %d"%best_epoch)
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/nlvr.yaml')
    parser.add_argument('--output_dir', default='output/NLVR')
    parser.add_argument('--evaluate', action='store_true')      
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--use_ceph', action='store_true')  
    parser.add_argument('--pretrained', default='pretrained/model_base_nlvr.pth', type=str)
    parser.add_argument('--w_sp_attn', default=4.8e-3, type=float, help='regularization coefficient for attn')
    parser.add_argument('--w_sp_mlp', default=2e-4, type=float, help='regularization coefficient for mlp')
    parser.add_argument('--epoch', default=15, type=int, help='number of epochs')
    parser.add_argument('--p', default=0.5, type=float, help='total compression ratio')  
    
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    if not args.use_ceph:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
        client=None
    else:
        client = Client('~/petreloss.conf', enable_mc=True)
        client.put(os.path.join('s3://BucketName/ProjectName', args.output_dir, 'config.yaml'), yaml.dump(config))

    main(args, config, client=client)
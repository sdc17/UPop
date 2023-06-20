import argparse
import os
from regex import B
import ruamel_yaml as yaml
import numpy as np
import random
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.blip_vqa import blip_vqa
import utils
from utils import cosine_lr_schedule, print_params_and_flops
from data import create_dataset, create_sampler, create_loader
from data.vqa_dataset import vqa_collate_fn
from data.utils import save_result


import io
# from petrel_client.client import Client
import math

from torch.cuda.amp import autocast as autocast

def update_alpha_parameters(model, layers, p, pi, print_info=True):

    standarlization = lambda x: (x - torch.mean(x)) / torch.std(x)
    alpha_grad_attn = torch.stack([
        torch.cat([getattr(model.module.visual_encoder.blocks, str(i)).attn.alpha.grad for i in range(layers)]),
        torch.stack([getattr(model.module.text_encoder.encoder.layer, str(i)).attention.self.alpha.grad for i in range(layers)]),
        torch.stack([getattr(model.module.text_encoder.encoder.layer, str(i)).crossattention.self.alpha.grad for i in range(layers)]),
        torch.stack([getattr(model.module.text_decoder.bert.encoder.layer, str(i)).attention.self.alpha.grad for i in range(layers)]),
        torch.stack([getattr(model.module.text_decoder.bert.encoder.layer, str(i)).crossattention.self.alpha.grad for i in range(layers)]),
    ])
    alpha_grad_mlp = torch.stack([
        torch.stack([getattr(model.module.visual_encoder.blocks, str(i)).mlp.alpha.grad for i in range(layers)]),
        torch.stack([getattr(model.module.text_encoder.encoder.layer, str(i)).intermediate.alpha.grad for i in range(layers)]),
        torch.stack([getattr(model.module.text_decoder.bert.encoder.layer, str(i)).intermediate.alpha.grad for i in range(layers)]),
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
        update(getattr(model.module.text_encoder.encoder.layer, str(i)).crossattention.self.alpha, alpha_grad_attn[2, i])
        update(getattr(model.module.text_decoder.bert.encoder.layer, str(i)).attention.self.alpha, alpha_grad_attn[3, i])
        update(getattr(model.module.text_decoder.bert.encoder.layer, str(i)).crossattention.self.alpha, alpha_grad_attn[4, i])
        update(getattr(model.module.visual_encoder.blocks, str(i)).mlp.alpha, alpha_grad_mlp[0, i])
        update(getattr(model.module.text_encoder.encoder.layer, str(i)).intermediate.alpha, alpha_grad_mlp[1, i])
        update(getattr(model.module.text_decoder.bert.encoder.layer, str(i)).intermediate.alpha, alpha_grad_mlp[2, i])

    if print_info:
        attn, mlp = [], []
        for i in range(layers):
            attn.append(getattr(model.module.visual_encoder.blocks, str(i)).attn.alpha.flatten())
            attn.append(getattr(model.module.text_encoder.encoder.layer, str(i)).attention.self.alpha.flatten())
            attn.append(getattr(model.module.text_encoder.encoder.layer, str(i)).crossattention.self.alpha.flatten())
            attn.append(getattr(model.module.text_decoder.bert.encoder.layer, str(i)).attention.self.alpha.flatten())
            attn.append(getattr(model.module.text_decoder.bert.encoder.layer, str(i)).crossattention.self.alpha.flatten())
            mlp.append(getattr(model.module.visual_encoder.blocks, str(i)).mlp.alpha.flatten())
            mlp.append(getattr(model.module.text_encoder.encoder.layer, str(i)).intermediate.alpha.flatten())
            mlp.append(getattr(model.module.text_decoder.bert.encoder.layer, str(i)).intermediate.alpha.flatten())
        print('Current compression ratio of attn: ', 1-torch.mean(torch.cat(attn)))
        print('Current compression ratio of mlp: ', 1-torch.mean(torch.cat(mlp)))
        print('Current compression ratio: ', pi)  



def train(model, data_loader, optimizer, epoch, device, config, search=False, scaler=None):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if search:
        metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_sp_attn', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_sp_mlp', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch) if not search else 'Search Epoch: [{}]'.format(epoch)

    print_freq = 50    
    len_data_loader = len(data_loader)
    total_steps = len_data_loader*config['max_epoch']

    for i,(image, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, weights = image.to(device,non_blocking=True), weights.to(device,non_blocking=True) 

        if scaler is not None:
            with autocast():   
                loss = model(image, question, answer, train=True, n=n, weights=weights)        
                if search:
                    sparsity_loss_attn, sparsity_loss_mlp = model.module.get_sparsity_loss()
                    metric_logger.update(loss_ce=loss.item()) 
                    metric_logger.update(loss_sp_attn=config['w_sp_attn'] * sparsity_loss_attn.item()) 
                    metric_logger.update(loss_sp_mlp=config['w_sp_mlp'] * sparsity_loss_mlp.item()) 
                    loss += config['w_sp_attn'] * sparsity_loss_attn + config['w_sp_mlp'] * sparsity_loss_mlp
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model(image, question, answer, train=True, n=n, weights=weights)        
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
        if search and (step % 1000 == 0 or step == total_steps - 1):
            pi = config['p']*((1-math.cos(math.pi*(step+1)/total_steps))/2)**(1/2)
            update_alpha_parameters(model, 12, config['p'], pi)

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 


@torch.no_grad()
def evaluation(model, data_loader, device, config) :
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    
    result = []
    
    if config['inference']=='rank':   
        answer_list = data_loader.dataset.answer_list
        answer_candidates = model.tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)    
        answer_candidates.input_ids[:,0] = model.tokenizer.bos_token_id
        
    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device,non_blocking=True)             

        if config['inference']=='generate':
            answers = model(image, question, train=False, inference='generate') 
            
            for answer, ques_id in zip(answers, question_id):
                ques_id = int(ques_id.item())       
                result.append({"question_id":ques_id, "answer":answer})             
            
        elif config['inference']=='rank':    
            answer_ids = model(image, question, answer_candidates, train=False, inference='rank', k_test=config['k_test'])      

            for ques_id, answer_id in zip(question_id, answer_ids):
                result.append({"question_id":int(ques_id.item()), "answer":answer_list[answer_id]})   

    return result


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
    print("Creating vqa datasets")
    datasets = create_dataset('vqa', config, client)   
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)         
    else:
        samplers = [None, None]
    
    train_loader, test_loader = create_loader(datasets,samplers,
                                              batch_size=[config['batch_size_train'],config['batch_size_test']],
                                              num_workers=[4,4],is_trains=[True, False], 
                                              collate_fns=[vqa_collate_fn,None]) 
    
    if not args.evaluate:
        print("Creating model for searching")
        search_model = blip_vqa(client=client, pretrained=config['pretrained'], image_size=config['image_size'], 
                        vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                        search=True)
        search_model = search_model.to(device)  
        print_params_and_flops('vqa', search_model, device)
        search_model_without_ddp = search_model
        if args.distributed:
            search_model = torch.nn.parallel.DistributedDataParallel(search_model, device_ids=[args.gpu])
            search_model_without_ddp = search_model.module    
        
        if not args.amp:
            optimizer = torch.optim.AdamW(
                    params=[{'params':[param for name, param in list(search_model.named_parameters()) if not ('alpha' in name)]}], 
                    lr=config['init_lr'], 
                    weight_decay=config['weight_decay']
                    )
        else:
            optimizer = torch.optim.AdamW(
                    [{'params':[param for name, param in list(search_model.named_parameters()) if not ('alpha' in name)], 
                      'lr': config['init_lr'], 'weight_decay': config['weight_decay']},
                     {'params':[param for name, param in list(search_model.named_parameters()) if ('alpha' in name)], 
                      'lr': 0, 'weight_decay': 0}]
                    )

        print("Start searching")
        scaler = torch.cuda.amp.GradScaler() if args.amp else None
        for epoch in range(0, config['max_epoch']):
            if args.evaluate:
                break
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train(search_model, train_loader, optimizer, epoch, device, config, search=True, scaler=scaler) 
        dist.barrier()   
        search_model.module.print_compression_statistics()

        #### Model #### 
        print("Creating model for training")
        model = blip_vqa(client=client, pretrained=config['pretrained'], image_size=config['image_size'], 
                        vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
        msg = model.load_state_dict(search_model_without_ddp.state_dict(), strict=False)
        model.compress(search_model_without_ddp)
    else:
        print("Creating model for evaluation")
        model = blip_vqa(client=client, pretrained='', image_size=config['image_size'], 
                        vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                        evaluate=True)
        model.prune_if_compressed(client, config['pretrained'])

    model = model.to(device)   
    print_params_and_flops('vqa', model, device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    best = 0
    best_epoch = 0 
       
    print("Start training")
    scaler = torch.cuda.amp.GradScaler() if (not args.evaluate and args.amp) else None
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
                
            train_stats = train(model, train_loader, optimizer, epoch, device, config, scaler=scaler) 

        else:         
            break        
        
        if utils.is_main_process():     
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                
            # with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            #     f.write(json.dumps(log_stats) + "\n")         
            print("LOG: ", log_stats)               
                    
            save_obj = {
                'model': model_without_ddp.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                # 'config': config,
                # 'epoch': epoch,
            }
            if client is not None:
                with io.BytesIO() as f:
                    torch.save(save_obj, f)
                    client.put(os.path.join('s3://BucketName/ProjectName', args.output_dir, 'checkpoint_%02d.pth'%epoch), f.getvalue())
            else:
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  

        dist.barrier()         

    vqa_result = evaluation(model_without_ddp, test_loader, device, config)        
    result_file = save_result(vqa_result, args.result_dir, 'vqa_result', client=client)  
                      
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/vqa.yaml') 
    parser.add_argument('--output_dir', default='output/VQA')
    parser.add_argument('--evaluate', action='store_true')      
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--use_ceph', action='store_true')  
    parser.add_argument('--pretrained', default='pretrained/model_base_vqa_capfilt_large.pth', type=str)
    parser.add_argument('--w_sp_attn', default=1.44e-2, type=float, help='regularization coefficient for attn')
    parser.add_argument('--w_sp_mlp', default=5e-4, type=float, help='regularization coefficient for mlp')
    parser.add_argument('--epoch', default=5, type=int, help='number of epoches')
    parser.add_argument('--p', default=0.5, type=float, help='total compression ratio')
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    if not args.use_ceph:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
        client=None
    else:
        client = Client('~/petreloss.conf', enable_mc=True)
        client.put(os.path.join('s3://BucketName/ProjectName', args.output_dir, 'config.yaml'), yaml.dump(config))

    main(args, config, client)
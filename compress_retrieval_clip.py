import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from clip import clip
import utils
from utils import cosine_lr_schedule, print_params_and_flops
from data import create_dataset, create_sampler, create_loader

import io
# from petrel_client.client import Client
import math

from torch.cuda.amp import autocast as autocast

def update_alpha_parameters(model, vision_layers, transformer_layers, p, pi, print_info=True):

    standarlization = lambda x, mean, std : (x - mean) / std

    alpha_grad_attn_vision = torch.stack([getattr(model.module.visual.transformer.resblocks, str(i)).attn.alpha.grad for i in range(vision_layers)])
    alpha_grad_attn_language = torch.stack([getattr(model.module.transformer.resblocks, str(i)).attn.alpha.grad for i in range(transformer_layers)])
    alpha_grad_attn = torch.cat([alpha_grad_attn_vision.view(-1), alpha_grad_attn_language.view(-1)])
    mean, std = torch.mean(alpha_grad_attn), torch.std(alpha_grad_attn)
    alpha_grad_attn_vision, alpha_grad_attn_language = standarlization(alpha_grad_attn_vision, mean, std), standarlization(alpha_grad_attn_language, mean, std)

    alpha_grad_mlp_vision = torch.stack([getattr(model.module.visual.transformer.resblocks, str(i)).alpha.grad for i in range(vision_layers)])
    alpha_grad_mlp_language = torch.stack([getattr(model.module.transformer.resblocks, str(i)).alpha.grad for i in range(transformer_layers)])
    alpha_grad_mlp = torch.cat([alpha_grad_mlp_vision.view(-1), alpha_grad_mlp_language.view(-1)])
    mean, std = torch.mean(alpha_grad_mlp), torch.std(alpha_grad_mlp)
    alpha_grad_mlp_vision, alpha_grad_mlp_language = standarlization(alpha_grad_mlp_vision, mean, std), standarlization(alpha_grad_mlp_language, mean, std)
    
    alpha_grad = torch.cat([alpha_grad_attn_vision.view(-1), alpha_grad_attn_language.view(-1), alpha_grad_mlp_vision.view(-1), alpha_grad_mlp_language.view(-1)])
    sorted_alpha_grad, indices = torch.sort(alpha_grad, descending=True)
    compression_weight = torch.ones_like(indices)
    compression_weight[indices < alpha_grad_attn.numel()] = 36
    threshold = sorted_alpha_grad[torch.argmin(torch.abs(torch.cumsum(compression_weight, 0) - torch.sum(compression_weight)*pi))]
    
    def update(module, grad):
        mask = ((grad <= threshold) | (grad <= torch.min(grad)))
        module.data.copy_(mask + (~mask)*(1 - pi/p))

    for i in range(vision_layers):
        update(getattr(model.module.visual.transformer.resblocks, str(i)).attn.alpha, alpha_grad_attn_vision[i])
        update(getattr(model.module.visual.transformer.resblocks, str(i)).alpha, alpha_grad_mlp_vision[i])
    for i in range(transformer_layers):
        update(getattr(model.module.transformer.resblocks, str(i)).attn.alpha, alpha_grad_attn_language[i])
        update(getattr(model.module.transformer.resblocks, str(i)).alpha, alpha_grad_mlp_language[i])

    if print_info:
        attn, mlp = [], []
        for i in range(vision_layers):
            attn.append(getattr(model.module.visual.transformer.resblocks, str(i)).attn.alpha.flatten())
            mlp.append(getattr(model.module.visual.transformer.resblocks, str(i)).alpha.flatten())
        for i in range(transformer_layers):
            attn.append(getattr(model.module.transformer.resblocks, str(i)).attn.alpha.flatten())
            mlp.append(getattr(model.module.transformer.resblocks, str(i)).alpha.flatten())
        print('Current compression ratio of attn: ', 1-torch.mean(torch.cat(attn)))
        print('Current compression ratio of mlp: ', 1-torch.mean(torch.cat(mlp)))
        print('Current compression ratio: ', pi)  


def train(model, data_loader, optimizer, epoch, device, config, search=False, interval=50, scaler=None):

    vision_layers, transformer_layers = model.module.vision_layers, model.module.transformer_layers
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if search:
        metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_sp_attn', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_sp_mlp', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch) if not search else 'Search Epoch: [{}]'.format(epoch)
    print_freq = 50
    len_data_loader = len(data_loader)
    total_steps = len_data_loader*config['max_epoch']

    for i,(image, caption, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device,non_blocking=True)   
        idx = idx.to(device,non_blocking=True)   

        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))
        
        if scaler is not None:
            with autocast():
                loss = model(image, caption, alpha=alpha, idx=idx)       
                if search:
                    sparsity_loss_attn, sparsity_loss_mlp = model.module.get_sparsity_loss()
                    metric_logger.update(loss_ita=loss.item()) 
                    metric_logger.update(loss_sp_attn=config['w_sp_attn'] * sparsity_loss_attn.item()) 
                    metric_logger.update(loss_sp_mlp=config['w_sp_mlp'] * sparsity_loss_mlp.item()) 
                    loss += config['w_sp_attn'] * sparsity_loss_attn + config['w_sp_mlp'] * sparsity_loss_mlp
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model(image, caption, alpha=alpha, idx=idx)       
            if search:
                sparsity_loss_attn, sparsity_loss_mlp = model.module.get_sparsity_loss()
                metric_logger.update(loss_ita=loss.item()) 
                metric_logger.update(loss_sp_attn=config['w_sp_attn'] * sparsity_loss_attn.item()) 
                metric_logger.update(loss_sp_mlp=config['w_sp_mlp'] * sparsity_loss_mlp.item()) 
                loss += config['w_sp_attn'] * sparsity_loss_attn + config['w_sp_mlp'] * sparsity_loss_mlp
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    

        step = epoch*len_data_loader+i
        if search and (step % interval == 0 or step == total_steps - 1):
            pi = config['p']*((1-math.cos(math.pi*(step+1)/total_steps))/2)**(1/2)
            update_alpha_parameters(model, vision_layers, transformer_layers, config['p'], pi)
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval() 
    
    print('Computing features for evaluation...')

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_embeds = []  
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = model.tokenize(text).to(device) 
        text_output = model.encode_text(text_input)
        text_embed = text_output / text_output.norm(dim=1, keepdim=True)
        text_embeds.append(text_embed)   
    text_embeds = torch.cat(text_embeds,dim=0)

    image_embeds = []
    for image, img_id in data_loader: 
        image = image.to(device) 
        image_feat = model.encode_image(image)
        image_embed = image_feat / image_feat.norm(dim=1, keepdim=True)
        image_embeds.append(image_embed)
    image_embeds = torch.cat(image_embeds,dim=0)

    sims_matrix = image_embeds @ text_embeds.t()
    return sims_matrix.cpu().numpy(), sims_matrix.t().cpu().numpy()

            
@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    for index,score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean}
    return eval_result


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
    config['init_lr'] = args.lr
    config['p'] = args.p
    if not args.evaluate:
        print('Target compression ratio: {}%'.format(config['p']*100))

    #### Dataset #### 
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('retrieval_%s'%config['dataset'], config, client)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False], 
                                                          collate_fns=[None,None,None])   


    if not args.evaluate:
        print("Creating model for searching")
        if client is not None:
            search_model, preprocess = clip.load_from_client(name=config['pretrained'], device=device, search=True, client=client)
        else:
            search_model, preprocess = clip.load(name=config['pretrained'], device=device, search=True)
        search_model.tokenize = clip.tokenize
        search_model.copy_params()
        print_params_and_flops('retrieval_clip', search_model, device, config)
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
        scaler = torch.cuda.amp.GradScaler() if args.amp else None
        for epoch in range(0, config['max_epoch']):
            if args.evaluate:
                break
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train(search_model, train_loader, optimizer, epoch, device, config, search=True, interval=50 if config['dataset']=='flickr' else 200, scaler=scaler) 
        dist.barrier()   
        search_model.module.print_compression_statistics()

        print("Creating model for training")
        if client is not None:
            model, preprocess = clip.load_from_client(name=config['pretrained'], device=device, client=client)
        else:
            model, preprocess = clip.load(name=config['pretrained'], device=device)
        msg = model.load_state_dict(search_model_without_ddp.state_dict(), strict=False)
        model.tokenize = clip.tokenize
        model.compress(search_model_without_ddp)
    else:
        print("Creating model for evaluation")
        if client is not None:
            model, preprocess = clip.load_from_client(name=config['pretrained'], device=device, client=client, evaluate=True)
        else:
            model, preprocess = clip.load(name=config['pretrained'], device=device, evaluate=True)
        model.tokenize = clip.tokenize
        model.prune_if_compressed(client, config['pretrained'])
        model = model.to(device)  

    model.copy_params()
    print_params_and_flops('retrieval_clip', model, device, config)
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

        score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, device, config)
        score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, device, config)
    
        if utils.is_main_process():  
      
            val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)  
            print(val_result)
                                
            if val_result['r_mean']>best:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                    # 'config': config,
                    # 'epoch': epoch,
                }
                if client is not None:
                    with io.BytesIO() as f:
                        torch.save(save_obj, f)
                        f.seek(0)
                        client.put(os.path.join('s3://BucketName/ProjectName', args.output_dir, 'checkpoint_best.pth'), f)
                else:
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))  
                best = val_result['r_mean']        
                best_epoch = epoch  
                
                test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt) 
                print(test_result)
            
            if args.evaluate:                
                log_stats = {**{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},                  
                            }
                # with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                #     f.write(json.dumps(log_stats) + "\n")     
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},  
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                            }
                # with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                #     f.write(json.dumps(log_stats) + "\n")   
            print("LOG: ", log_stats)
                    
        if args.evaluate: 
            break

        dist.barrier()     
        torch.cuda.empty_cache()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/retrieval_flickr.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_flickr')        
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--use_ceph', action='store_true')  
    parser.add_argument('--pretrained', default='pretrained/clip_large_retrieval_flickr.pth', type=str)
    parser.add_argument('--w_sp_attn', default=(22/15)*8e-3, type=float, help='regularization coefficient for attn')
    parser.add_argument('--w_sp_mlp', default=2e-4, type=float, help='regularization coefficient for mlp')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--epoch', default=12, type=int, help='number of epoches')
    parser.add_argument('--p', default=0.5, type=float, help='total compression ratio')
    parser.add_argument('--amp', action='store_true')
    
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    if not args.use_ceph:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
        client=None
    else:
        client = Client('~/petreloss.conf', enable_mc=True)
        client.put(os.path.join('s3://BucketName/ProjectName', args.output_dir, 'config.yaml'), yaml.dump(config))
    
    main(args, config, client)
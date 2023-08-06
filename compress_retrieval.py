'''
 * Copyright (c) 2023, Dachuan Shi.
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * For full license text, see LICENSE.txt file in the repo root
 * By Dachuan Shi
'''
import argparse
import os
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

from models.blip_retrieval import blip_retrieval
import utils
from utils import cosine_lr_schedule, print_params_and_flops
from data import create_dataset, create_sampler, create_loader

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
    ])
    alpha_grad_mlp = torch.stack([
        torch.stack([getattr(model.module.visual_encoder.blocks, str(i)).mlp.alpha.grad for i in range(layers)]),
        torch.stack([getattr(model.module.text_encoder.encoder.layer, str(i)).intermediate.alpha.grad for i in range(layers)]),
    ])
    alpha_grad_attn, alpha_grad_mlp = standarlization(alpha_grad_attn), standarlization(alpha_grad_mlp)
    alpha_grad = torch.cat([alpha_grad_attn.view(-1), alpha_grad_mlp.view(-1)])
    sorted_alpha_grad, indices = torch.sort(alpha_grad, descending=True)
    compression_weight = torch.ones_like(indices)
    compression_weight[indices < alpha_grad_attn.numel()] = 36 # 36 = 12 (number of heads) * [1 (weights of query) + 1 (weights of key) + 1 (weights of value)]
    threshold = sorted_alpha_grad[torch.argmin(torch.abs(torch.cumsum(compression_weight, 0) - torch.sum(compression_weight)*pi))]
    
    def update(module, grad):
        mask = ((grad <= threshold) | (grad <= torch.min(grad)))
        module.data.copy_(mask + (~mask)*(1 - pi/p))

    for i in range(layers):
        update(getattr(model.module.visual_encoder.blocks, str(i)).attn.alpha, alpha_grad_attn[0, i].unsqueeze(0))
        update(getattr(model.module.text_encoder.encoder.layer, str(i)).attention.self.alpha, alpha_grad_attn[1, i])
        update(getattr(model.module.text_encoder.encoder.layer, str(i)).crossattention.self.alpha, alpha_grad_attn[2, i])
        update(getattr(model.module.visual_encoder.blocks, str(i)).mlp.alpha, alpha_grad_mlp[0, i])
        update(getattr(model.module.text_encoder.encoder.layer, str(i)).intermediate.alpha, alpha_grad_mlp[1, i])

    if print_info:
        attn, mlp = [], []
        for i in range(layers):
            attn.append(getattr(model.module.visual_encoder.blocks, str(i)).attn.alpha.flatten())
            attn.append(getattr(model.module.text_encoder.encoder.layer, str(i)).attention.self.alpha.flatten())
            attn.append(getattr(model.module.text_encoder.encoder.layer, str(i)).crossattention.self.alpha.flatten())
            mlp.append(getattr(model.module.visual_encoder.blocks, str(i)).mlp.alpha.flatten())
            mlp.append(getattr(model.module.text_encoder.encoder.layer, str(i)).intermediate.alpha.flatten())
        print('Current compression ratio of attn: ', 1-torch.mean(torch.cat(attn)))
        print('Current compression ratio of mlp: ', 1-torch.mean(torch.cat(mlp)))
        print('Current compression ratio: ', pi)  


def train(model, data_loader, optimizer, epoch, device, config, search=False, scaler=None):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if search:
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
                loss_ita, loss_itm = model(image, caption, alpha=alpha, idx=idx)                  
                loss = loss_ita + loss_itm
                if search:
                    sparsity_loss_attn, sparsity_loss_mlp = model.module.get_sparsity_loss()
                    metric_logger.update(loss_sp_attn=config['w_sp_attn'] * sparsity_loss_attn.item()) 
                    metric_logger.update(loss_sp_mlp=config['w_sp_mlp'] * sparsity_loss_mlp.item()) 
                    loss += config['w_sp_attn'] * sparsity_loss_attn + config['w_sp_mlp'] * sparsity_loss_mlp
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_ita, loss_itm = model(image, caption, alpha=alpha, idx=idx)                  
            loss = loss_ita + loss_itm
            if search:
                sparsity_loss_attn, sparsity_loss_mlp = model.module.get_sparsity_loss()
                metric_logger.update(loss_sp_attn=config['w_sp_attn'] * sparsity_loss_attn.item()) 
                metric_logger.update(loss_sp_mlp=config['w_sp_mlp'] * sparsity_loss_mlp.item()) 
                loss += config['w_sp_attn'] * sparsity_loss_attn + config['w_sp_mlp'] * sparsity_loss_mlp
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
        
        step = epoch*len_data_loader+i
        if search and (step % 200 == 0 or step == total_steps - 1):
            pi = config['p']*((1-math.cos(math.pi*(step+1)/total_steps))/2)**(1/2)
            update_alpha_parameters(model, 12 if config['vit']=='base' else 24, config['p'], pi)

        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  

@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []  
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
        text_embeds.append(text_embed)   
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)
    
    text_embeds = torch.cat(text_embeds,dim=0)
    text_ids = torch.cat(text_ids,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    text_ids[:,0] = model.tokenizer.enc_token_id
    
    image_feats = []
    image_embeds = []
    for image, img_id in data_loader: 
        image = image.to(device) 
        image_feat = model.visual_encoder(image)   
        image_embed = model.vision_proj(image_feat[:,0,:])            
        image_embed = F.normalize(image_embed,dim=-1)      
        
        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)
     
    image_feats = torch.cat(image_feats,dim=0)
    image_embeds = torch.cat(image_embeds,dim=0)
    
    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full((len(data_loader.dataset.image),len(texts)),-100.0).to(device)
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)

    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_feats[start+i].repeat(config['k_test'],1,1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[topk_idx], 
                                    attention_mask = text_atts[topk_idx],
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_i2t[start+i,topk_idx] = score + topk_sim
        
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)    
    
    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[start+i].repeat(config['k_test'],1), 
                                    attention_mask = text_atts[start+i].repeat(config['k_test'],1),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[start+i,topk_idx] = score + topk_sim

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


            
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
        print("Test parameters and FLOPs")
        search_model = blip_retrieval(client=client, pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                             vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                             queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'],
                             search=True)
        search_model = search_model.to(device)  
        print_params_and_flops('retrieval', search_model, device)

        print("Creating model for searching ")
        search_model = blip_retrieval(client=client, pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                                vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                                queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'],
                                search=True)
        search_model = search_model.to(device)  
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

        print("Test parameters and FLOPs")
        model = blip_retrieval(client=client, pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                                    vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                                    queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])
        msg = model.load_state_dict(search_model_without_ddp.state_dict(), strict=False)
        model.compress(search_model_without_ddp)
        model = model.to(device)   
        print_params_and_flops('retrieval', model, device)

        print("Creating model for training")
        model = blip_retrieval(client=client, pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                                vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                                queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])
        msg = model.load_state_dict(search_model_without_ddp.state_dict(), strict=False)
        model.compress(search_model_without_ddp)
    else:
        print("Test parameters and FLOPs")
        model = blip_retrieval(client=client, pretrained='', image_size=config['image_size'], vit=config['vit'], 
                                vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                                queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'],
                                evaluate=True)
        model.prune_if_compressed(client, config['pretrained'])
        model = model.to(device)   
        print_params_and_flops('retrieval', model, device)

        print("Creating model for evaluation")
        model = blip_retrieval(client=client, pretrained='', image_size=config['image_size'], vit=config['vit'], 
                                vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                                queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'],
                                evaluate=True)
        model.prune_if_compressed(client, config['pretrained'])

    model.copy_params()
    model = model.to(device)   
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
        
        if epoch >= (config['max_epoch']//2) or args.evaluate:
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
                            client.put(os.path.join('s3://BucketName/ProjectName', args.output_dir, 'checkpoint_best.pth'), f.getvalue())
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
    parser.add_argument('--pretrained', default='pretrained/model_base_retrieval_coco.pth', type=str)
    parser.add_argument('--w_sp_attn', default=6.4e-3, type=float, help='regularization coefficient for attn')
    parser.add_argument('--w_sp_mlp', default=2e-4, type=float, help='regularization coefficient for mlp')
    parser.add_argument('--epoch', default=15, type=int, help='number of epoches')
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
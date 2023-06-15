from telnetlib import PRAGMA_HEARTBEAT
from models.med import BertConfig
from models.nlvr_encoder import BertModel
from models.vit import interpolate_pos_embed
from models.blip import create_vit, init_tokenizer, is_url

from timm.models.hub import download_cached_file

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
import os
import io
import functools

class BLIP_NLVR(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 480,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                   
                 search=False,
                 evaluate=False
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        self.layers = 12 if vit == 'base' else 24
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1, search=search, evaluate=evaluate)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        med_config.search = search
        med_config.evaluate = evaluate
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False) 

        self.cls_head = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, 2)
                )  


    def forward(self, image, text, targets, train=True):
        image_embeds = self.visual_encoder(image) 

        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        image0_embeds, image1_embeds = torch.split(image_embeds,targets.size(0))     
        text = self.tokenizer(text, padding='longest', return_tensors="pt").to(image.device) 
        text.input_ids[:,0] = self.tokenizer.enc_token_id        

        output = self.text_encoder(text.input_ids, 
                                   attention_mask = text.attention_mask, 
                                   encoder_hidden_states = [image0_embeds,image1_embeds],
                                   encoder_attention_mask = [image_atts[:image0_embeds.size(0)],
                                                             image_atts[image0_embeds.size(0):]],        
                                   return_dict = True,
                                  )  
        hidden_state = output.last_hidden_state[:,0,:]        
        prediction = self.cls_head(hidden_state)

        if train:            
            loss = F.cross_entropy(prediction, targets)   
            return loss
        else:
            return prediction

    
    def forward_throughput(self, image, text, targets):
        
        image_embeds = self.visual_encoder(image) 

        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device) 
        image0_embeds, image1_embeds = torch.split(image_embeds,targets.size(0))     

        output = self.text_encoder(text.input_ids, 
                                   attention_mask = text.attention_mask, 
                                   encoder_hidden_states = [image0_embeds,image1_embeds],
                                   encoder_attention_mask = [image_atts[:image0_embeds.size(0)],
                                                             image_atts[image0_embeds.size(0):]],        
                                   return_dict = True,
                                  )  
        hidden_state = output.last_hidden_state[:,0,:]        
        prediction = self.cls_head(hidden_state)

        return prediction


    def get_sparsity_loss(self):
        sparsity_loss_attn, sparsity_loss_mlp = 0, 0
        for i in range(self.layers):
            sparsity_loss_attn += torch.sum(torch.abs(getattr(self.visual_encoder.blocks, str(i)).attn.alpha))
            sparsity_loss_mlp += torch.sum(torch.abs(getattr(self.visual_encoder.blocks, str(i)).mlp.alpha))
            sparsity_loss_attn += torch.sum(torch.abs(getattr(self.text_encoder.encoder.layer, str(i)).attention.self.alpha))
            sparsity_loss_attn += torch.sum(torch.abs(getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.alpha))
            sparsity_loss_attn += torch.sum(torch.abs(getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.alpha))
            sparsity_loss_mlp += torch.sum(torch.abs(getattr(self.text_encoder.encoder.layer, str(i)).intermediate.alpha))
        return sparsity_loss_attn, sparsity_loss_mlp

        
    def compress(self, search_model):

        for i in range(self.layers):
            # vit mlp
            in_features = getattr(self.visual_encoder.blocks, str(i)).mlp.fc1.weight.shape[-1]
            out_features = getattr(self.visual_encoder.blocks, str(i)).mlp.fc2.weight.shape[0]
            alpha = torch.squeeze(getattr(search_model.visual_encoder.blocks, str(i)).mlp.alpha.data)
            hidden_features = torch.count_nonzero(alpha)
            getattr(self.visual_encoder.blocks, str(i)).mlp.fc1 = nn.Linear(in_features, hidden_features)
            getattr(self.visual_encoder.blocks, str(i)).mlp.fc1.weight.data = getattr(search_model.visual_encoder.blocks, str(i)).mlp.fc1.weight.data[alpha==1,:]
            getattr(self.visual_encoder.blocks, str(i)).mlp.fc1.bias.data = getattr(search_model.visual_encoder.blocks, str(i)).mlp.fc1.bias.data[alpha==1]
            getattr(self.visual_encoder.blocks, str(i)).mlp.fc2 = nn.Linear(hidden_features, out_features)
            getattr(self.visual_encoder.blocks, str(i)).mlp.fc2.weight.data = getattr(search_model.visual_encoder.blocks, str(i)).mlp.fc2.weight.data[:, alpha==1]
            getattr(self.visual_encoder.blocks, str(i)).mlp.fc2.bias.data = getattr(search_model.visual_encoder.blocks, str(i)).mlp.fc2.bias.data

            # vit attn
            in_features = getattr(self.visual_encoder.blocks, str(i)).attn.qkv.weight.shape[-1]
            out_features = getattr(self.visual_encoder.blocks, str(i)).attn.proj.weight.shape[0]
            alpha = torch.squeeze(getattr(search_model.visual_encoder.blocks, str(i)).attn.alpha.data)
            hidden_features = torch.count_nonzero(alpha)
            parameter_ratio = 3*getattr(self.visual_encoder.blocks, str(i)).attn.num_heads
            getattr(self.visual_encoder.blocks, str(i)).attn.qkv = nn.Linear(in_features, hidden_features*parameter_ratio)
            getattr(self.visual_encoder.blocks, str(i)).attn.qkv.weight.data = \
                getattr(search_model.visual_encoder.blocks, str(i)).attn.qkv.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.visual_encoder.blocks, str(i)).attn.qkv.weight.bias = \
                getattr(search_model.visual_encoder.blocks, str(i)).attn.qkv.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.visual_encoder.blocks, str(i)).attn.proj = nn.Linear(hidden_features, out_features)
            getattr(self.visual_encoder.blocks, str(i)).attn.proj.weight.data = \
                getattr(search_model.visual_encoder.blocks, str(i)).attn.proj.weight.data[:, alpha.repeat(parameter_ratio//3)==1]
            getattr(self.visual_encoder.blocks, str(i)).attn.proj.bias.data = getattr(search_model.visual_encoder.blocks, str(i)).attn.proj.bias.data
            
            # bert mlp
            in_features = getattr(self.text_encoder.encoder.layer, str(i)).intermediate.dense.weight.shape[-1]
            out_features = getattr(self.text_encoder.encoder.layer, str(i)).output.dense.weight.shape[0]
            alpha = torch.squeeze(getattr(search_model.text_encoder.encoder.layer, str(i)).intermediate.alpha.data)
            hidden_features = torch.count_nonzero(alpha)
            getattr(self.text_encoder.encoder.layer, str(i)).intermediate.dense = nn.Linear(in_features, hidden_features)
            getattr(self.text_encoder.encoder.layer, str(i)).intermediate.dense.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).intermediate.dense.weight.data[alpha==1,:]
            getattr(self.text_encoder.encoder.layer, str(i)).intermediate.dense.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).intermediate.dense.bias.data[alpha==1]
            getattr(self.text_encoder.encoder.layer, str(i)).output.dense = nn.Linear(hidden_features, out_features)
            getattr(self.text_encoder.encoder.layer, str(i)).output.dense.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).output.dense.weight.data[:, alpha==1]
            getattr(self.text_encoder.encoder.layer, str(i)).output.dense.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).output.dense.bias.data

            # bert attn
            in_features_query = getattr(self.text_encoder.encoder.layer, str(i)).attention.self.query.weight.shape[-1]
            in_features_key = getattr(self.text_encoder.encoder.layer, str(i)).attention.self.key.weight.shape[-1]
            in_features_value = getattr(self.text_encoder.encoder.layer, str(i)).attention.self.value.weight.shape[-1]
            out_features = getattr(self.text_encoder.encoder.layer, str(i)).attention.output.dense.weight.shape[0]
            alpha = torch.squeeze(getattr(search_model.text_encoder.encoder.layer, str(i)).attention.self.alpha.data)
            hidden_features = torch.count_nonzero(alpha)
            getattr(self.text_encoder.encoder.layer, str(i)).attention.self.attention_head_size = hidden_features
            getattr(self.text_encoder.encoder.layer, str(i)).attention.self.all_head_size = \
                hidden_features * getattr(self.text_encoder.encoder.layer, str(i)).attention.self.num_attention_heads
            parameter_ratio = getattr(self.text_encoder.encoder.layer, str(i)).attention.self.num_attention_heads
            getattr(self.text_encoder.encoder.layer, str(i)).attention.self.query = nn.Linear(in_features_query, hidden_features)
            getattr(self.text_encoder.encoder.layer, str(i)).attention.self.query.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).attention.self.query.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.text_encoder.encoder.layer, str(i)).attention.self.query.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).attention.self.query.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.text_encoder.encoder.layer, str(i)).attention.self.key = nn.Linear(in_features_key, hidden_features)
            getattr(self.text_encoder.encoder.layer, str(i)).attention.self.key.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).attention.self.key.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.text_encoder.encoder.layer, str(i)).attention.self.key.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).attention.self.key.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.text_encoder.encoder.layer, str(i)).attention.self.value = nn.Linear(in_features_value, hidden_features)
            getattr(self.text_encoder.encoder.layer, str(i)).attention.self.value.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).attention.self.value.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.text_encoder.encoder.layer, str(i)).attention.self.value.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).attention.self.value.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.text_encoder.encoder.layer, str(i)).attention.output.dense = nn.Linear(hidden_features, out_features)
            getattr(self.text_encoder.encoder.layer, str(i)).attention.output.dense.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).attention.output.dense.weight.data[:, alpha.repeat(parameter_ratio)==1]
            getattr(self.text_encoder.encoder.layer, str(i)).attention.output.dense.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).attention.output.dense.bias.data
        
            # corss att 0
            in_features_query = getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.query.weight.shape[-1]
            in_features_key = getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.key.weight.shape[-1]
            in_features_value = getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.value.weight.shape[-1]
            out_features = getattr(self.text_encoder.encoder.layer, str(i)).crossattention.output.dense0.weight.shape[0]
            alpha = torch.squeeze(getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self0.alpha.data)
            hidden_features = torch.count_nonzero(alpha)
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.attention_head_size = hidden_features
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.all_head_size = \
                hidden_features * getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.num_attention_heads
            parameter_ratio = getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.num_attention_heads
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.query = nn.Linear(in_features_query, hidden_features)
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.query.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self0.query.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.query.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self0.query.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.key = nn.Linear(in_features_key, hidden_features)
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.key.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self0.key.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.key.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self0.key.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.value = nn.Linear(in_features_value, hidden_features)
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.value.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self0.value.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.value.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self0.value.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.output.dense0 = nn.Linear(hidden_features, out_features)
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.output.dense0.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.output.dense0.weight.data[:, alpha.repeat(parameter_ratio)==1]
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.output.dense0.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.output.dense0.bias.data

            # corss att 1
            in_features_query = getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.query.weight.shape[-1]
            in_features_key = getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.key.weight.shape[-1]
            in_features_value = getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.value.weight.shape[-1]
            out_features = getattr(self.text_encoder.encoder.layer, str(i)).crossattention.output.dense1.weight.shape[0]
            alpha = torch.squeeze(getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self1.alpha.data)
            hidden_features = torch.count_nonzero(alpha)
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.attention_head_size = hidden_features
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.all_head_size = \
                hidden_features * getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.num_attention_heads
            parameter_ratio = getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.num_attention_heads
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.query = nn.Linear(in_features_query, hidden_features)
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.query.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self1.query.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.query.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self1.query.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.key = nn.Linear(in_features_key, hidden_features)
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.key.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self1.key.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.key.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self1.key.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.value = nn.Linear(in_features_value, hidden_features)
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.value.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self1.value.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.value.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self1.value.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.output.dense1 = nn.Linear(hidden_features, out_features)
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.output.dense1.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.output.dense1.weight.data[:, alpha.repeat(parameter_ratio)==1]
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.output.dense1.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.output.dense1.bias.data


    def print_compression_statistics(self):
        mask_attn_vision_list, mask_attn_language_list  = [], []
        mask_cross_attn0_list, mask_cross_attn1_list = [], []
        mask_mlp_vision_list, mask_mlp_language_list = [], []
        reserved_ratio = lambda x: (torch.count_nonzero(x) / torch.numel(x)).item()
        for i in range(self.layers):
            mask_attn_vision_list.append(getattr(self.visual_encoder.blocks, str(i)).attn.alpha.data.view(-1))
            mask_mlp_vision_list.append(getattr(self.visual_encoder.blocks, str(i)).mlp.alpha.data.view(-1))
            mask_attn_language_list.append(getattr(self.text_encoder.encoder.layer, str(i)).attention.self.alpha.data.view(-1))
            mask_mlp_language_list.append(getattr(self.text_encoder.encoder.layer, str(i)).intermediate.alpha.data.view(-1))
            mask_cross_attn0_list.append(getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.alpha.data.view(-1))
            mask_cross_attn1_list.append(getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.alpha.data.view(-1))
        print_format = lambda x: [round(i * 100, 2) for i in x]
        print('mask_attn_vision:  ', print_format([reserved_ratio(x) for x in mask_attn_vision_list]))
        print('mask_attn_language: ', print_format([reserved_ratio(x) for x in mask_attn_language_list]))
        print('mask_cross_attn0: ', print_format([reserved_ratio(x) for x in mask_cross_attn0_list]))
        print('mask_cross_attn1: ', print_format([reserved_ratio(x) for x in mask_cross_attn1_list]))
        print('mask_mlp_vision: ', print_format([reserved_ratio(x) for x in mask_mlp_vision_list]))
        print('mask_mlp_language: ', print_format([reserved_ratio(x) for x in mask_mlp_language_list]))
        print('mask_vision: ', reserved_ratio(torch.cat(mask_attn_vision_list + mask_mlp_vision_list)))
        print('mask_language: ', reserved_ratio(torch.cat(mask_attn_language_list + mask_mlp_language_list)))
        print('mask_cross_attn: ', reserved_ratio(torch.cat(mask_cross_attn0_list + mask_cross_attn1_list)))
        print('mask_attn: ', reserved_ratio(torch.cat(mask_attn_vision_list + mask_attn_language_list + mask_cross_attn0_list + mask_cross_attn1_list)))
        print('mask_mlp: ', reserved_ratio(torch.cat(mask_mlp_vision_list + mask_mlp_language_list)))


    def prune_if_compressed(self, client, url_or_filename):
        
        if client is not None:
            with io.BytesIO(client.get(os.path.join('s3://sdcBucket/BLIP-main', url_or_filename), enable_cache=True)) as f:
                checkpoint = torch.load(f, map_location='cpu')
        elif is_url(url_or_filename):
            cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
            checkpoint = torch.load(cached_file, map_location='cpu') 
        elif os.path.isfile(url_or_filename):        
            checkpoint = torch.load(url_or_filename, map_location='cpu') 
        else:
            raise RuntimeError('checkpoint url or path is invalid')
        state_dict = checkpoint['model']
        
        for i in range(self.layers):
            # vit mlp
            if getattr(self.visual_encoder.blocks, str(i)).mlp.fc1.weight.shape != state_dict['visual_encoder.blocks.'+str(i)+'.mlp.fc1.weight'].shape:
                del getattr(self.visual_encoder.blocks, str(i)).mlp.fc1
                getattr(self.visual_encoder.blocks, str(i)).mlp.fc1 = nn.Linear(*state_dict['visual_encoder.blocks.'+str(i)+'.mlp.fc1.weight'].shape[::-1])
                del getattr(self.visual_encoder.blocks, str(i)).mlp.fc2
                getattr(self.visual_encoder.blocks, str(i)).mlp.fc2 = nn.Linear(*state_dict['visual_encoder.blocks.'+str(i)+'.mlp.fc2.weight'].shape[::-1])

            # vit attn
            if getattr(self.visual_encoder.blocks, str(i)).attn.qkv.weight.shape != state_dict['visual_encoder.blocks.'+str(i)+'.attn.qkv.weight'].shape:
                del getattr(self.visual_encoder.blocks, str(i)).attn.qkv
                getattr(self.visual_encoder.blocks, str(i)).attn.qkv = nn.Linear(*state_dict['visual_encoder.blocks.'+str(i)+'.attn.qkv.weight'].shape[::-1])
                del getattr(self.visual_encoder.blocks, str(i)).attn.proj
                getattr(self.visual_encoder.blocks, str(i)).attn.proj = nn.Linear(*state_dict['visual_encoder.blocks.'+str(i)+'.attn.proj.weight'].shape[::-1])
            
            # bert mlp
            if getattr(self.text_encoder.encoder.layer, str(i)).intermediate.dense.weight.shape != state_dict['text_encoder.encoder.layer.'+str(i)+'.intermediate.dense.weight'].shape:
                del getattr(self.text_encoder.encoder.layer, str(i)).intermediate.dense
                getattr(self.text_encoder.encoder.layer, str(i)).intermediate.dense = nn.Linear(*state_dict['text_encoder.encoder.layer.'+str(i)+'.intermediate.dense.weight'].shape[::-1])
                del getattr(self.text_encoder.encoder.layer, str(i)).output.dense
                getattr(self.text_encoder.encoder.layer, str(i)).output.dense = nn.Linear(*state_dict['text_encoder.encoder.layer.'+str(i)+'.output.dense.weight'].shape[::-1])
            
            # bert attn
            if getattr(self.text_encoder.encoder.layer, str(i)).attention.self.query.weight.shape != state_dict['text_encoder.encoder.layer.'+str(i)+'.attention.self.query.weight'].shape:
                getattr(self.text_encoder.encoder.layer, str(i)).attention.self.attention_head_size = state_dict['text_encoder.encoder.layer.'+str(i)+'.attention.self.query.weight'].shape[-2] // \
                    getattr(self.text_encoder.encoder.layer, str(i)).attention.self.num_attention_heads
                getattr(self.text_encoder.encoder.layer, str(i)).attention.self.all_head_size = getattr(self.text_encoder.encoder.layer, str(i)).attention.self.attention_head_size * \
                    getattr(self.text_encoder.encoder.layer, str(i)).attention.self.num_attention_heads
                del getattr(self.text_encoder.encoder.layer, str(i)).attention.self.query
                getattr(self.text_encoder.encoder.layer, str(i)).attention.self.query = nn.Linear(*state_dict['text_encoder.encoder.layer.'+str(i)+'.attention.self.query.weight'].shape[::-1])
                del getattr(self.text_encoder.encoder.layer, str(i)).attention.self.key
                getattr(self.text_encoder.encoder.layer, str(i)).attention.self.key = nn.Linear(*state_dict['text_encoder.encoder.layer.'+str(i)+'.attention.self.key.weight'].shape[::-1])
                del getattr(self.text_encoder.encoder.layer, str(i)).attention.self.value
                getattr(self.text_encoder.encoder.layer, str(i)).attention.self.value = nn.Linear(*state_dict['text_encoder.encoder.layer.'+str(i)+'.attention.self.value.weight'].shape[::-1])
                del getattr(self.text_encoder.encoder.layer, str(i)).attention.output.dense
                getattr(self.text_encoder.encoder.layer, str(i)).attention.output.dense = nn.Linear(*state_dict['text_encoder.encoder.layer.'+str(i)+'.attention.output.dense.weight'].shape[::-1])
        
            # corss att 0
            if getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.query.weight.shape != state_dict['text_encoder.encoder.layer.'+str(i)+'.crossattention.self0.query.weight'].shape:
                getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.attention_head_size = state_dict['text_encoder.encoder.layer.'+str(i)+'.crossattention.self0.query.weight'].shape[-2] // \
                    getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.num_attention_heads
                getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.all_head_size = getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.attention_head_size * \
                    getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.num_attention_heads
                del getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.query
                getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.query = nn.Linear(*state_dict['text_encoder.encoder.layer.'+str(i)+'.crossattention.self0.query.weight'].shape[::-1])
                del getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.key
                getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.key = nn.Linear(*state_dict['text_encoder.encoder.layer.'+str(i)+'.crossattention.self0.key.weight'].shape[::-1])
                del getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.value
                getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self0.value = nn.Linear(*state_dict['text_encoder.encoder.layer.'+str(i)+'.crossattention.self0.value.weight'].shape[::-1])
                del getattr(self.text_encoder.encoder.layer, str(i)).crossattention.output.dense0
                getattr(self.text_encoder.encoder.layer, str(i)).crossattention.output.dense0 = nn.Linear(*state_dict['text_encoder.encoder.layer.'+str(i)+'.crossattention.output.dense0.weight'].shape[::-1])

            # corss att 1
            if getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.query.weight.shape != state_dict['text_encoder.encoder.layer.'+str(i)+'.crossattention.self1.query.weight'].shape:
                getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.attention_head_size = state_dict['text_encoder.encoder.layer.'+str(i)+'.crossattention.self1.query.weight'].shape[-2] // \
                    getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.num_attention_heads
                getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.all_head_size = getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.attention_head_size * \
                    getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.num_attention_heads
                del getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.query
                getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.query = nn.Linear(*state_dict['text_encoder.encoder.layer.'+str(i)+'.crossattention.self1.query.weight'].shape[::-1])
                del getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.key
                getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.key = nn.Linear(*state_dict['text_encoder.encoder.layer.'+str(i)+'.crossattention.self1.key.weight'].shape[::-1])
                del getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.value
                getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self1.value = nn.Linear(*state_dict['text_encoder.encoder.layer.'+str(i)+'.crossattention.self1.value.weight'].shape[::-1])
                del getattr(self.text_encoder.encoder.layer, str(i)).crossattention.output.dense1
                getattr(self.text_encoder.encoder.layer, str(i)).crossattention.output.dense1 = nn.Linear(*state_dict['text_encoder.encoder.layer.'+str(i)+'.crossattention.output.dense1.weight'].shape[::-1])

        # torch.cuda.empty_cache()
        self.load_state_dict(state_dict, strict=False)

        
def blip_nlvr(client, pretrained='',**kwargs):
    model = BLIP_NLVR(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained, client)
        print("missing keys:")
        print(msg.missing_keys)
    return model  

    
def load_checkpoint(model, url_or_filename, client=None):
    if client is not None:
        with io.BytesIO(client.get(os.path.join('s3://sdcBucket/BLIP-main', url_or_filename), enable_cache=True)) as f:
            checkpoint = torch.load(f, map_location='cpu')
    elif is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
    state_dict = checkpoint['model']
    
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
    
    for key in list(state_dict.keys()):
        if 'crossattention.self.' in key:
            new_key0 = key.replace('self','self0')
            new_key1 = key.replace('self','self1')
            state_dict[new_key0] = state_dict[key]
            state_dict[new_key1] = state_dict[key]
        elif 'crossattention.output.dense.' in key:
            new_key0 = key.replace('dense','dense0')
            new_key1 = key.replace('dense','dense1')
            state_dict[new_key0] = state_dict[key]
            state_dict[new_key1] = state_dict[key]  

    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % url_or_filename)
    return (model, msg)


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))
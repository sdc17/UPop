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
from functools import reduce 

class BLIP_NLVR(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 480,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                   
                 embeddings=None, 
                 layers=None,
                 search=False,
                 kd=False
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        self.layers = layers
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, 0.1, embeddings, layers, search=search)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        med_config.search = search
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False) 

        self.cls_head = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, 2)
                )  

        if kd:
            self.cross_kd_i2t = nn.Sequential(
                  nn.Linear(vision_width, 4*vision_width),
                  nn.ReLU(),
                  nn.Linear(4*vision_width, med_config.encoder_width)
                )  

            self.cross_kd_t2i_0 = nn.Sequential(
                  nn.Linear(med_config.encoder_width, 4*med_config.encoder_width),
                  nn.ReLU(),
                  nn.Linear(4*med_config.encoder_width, vision_width)
                )  

            self.cross_kd_t2i_1 = nn.Sequential(
                  nn.Linear(med_config.encoder_width, 4*med_config.encoder_width),
                  nn.ReLU(),
                  nn.Linear(4*med_config.encoder_width, vision_width)
                )  


    def forward(self, image, text, targets, train=True, kd=False):
        
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

        # if train:            
        #     loss = F.cross_entropy(prediction, targets)  
        #     return (loss, prediction, image_embeds, output.last_hidden_state, self.cross_kd_i2t(image0_embeds)[:,0,:], self.cross_kd_i2t(image1_embeds)[:,0,:], \
        #         self.cross_kd_t2i_0(output.last_hidden_state)[:,0,:], self.cross_kd_t2i_1(output.last_hidden_state)[:,0,:]) if kd else loss
        # else:
        #     return (prediction, image_embeds, output.last_hidden_state, image0_embeds, image1_embeds) if kd else prediction

        # if train:            
        #     loss = F.cross_entropy(prediction, targets)  
        #     return (loss, image_embeds.shape[1], image0_embeds[:,0,:], image1_embeds[:,0,:], hidden_state, self.cross_kd_i2t(image0_embeds[:,0,:]), self.cross_kd_i2t(image1_embeds[:,0,:]), \
        #         self.cross_kd_t2i_0(hidden_state), self.cross_kd_t2i_1(hidden_state)) if kd else loss
        # else:
        #     return prediction

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
        

def blip_nlvr(client, pretrained='',**kwargs):
    model = BLIP_NLVR(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained, kwargs['vit'] == 'custom', client)
        print("missing keys:")
        print(msg.missing_keys)
    return model  

    
def load_checkpoint(model, url_or_filename, load_from_different_size=True, client=None):
    if client is not None:
        # with io.BytesIO(client.get(os.path.join('s3://sdcBucket/BLIP-main', url_or_filename), enable_cache=True)) as f:
        with io.BytesIO(client.get(os.path.join('s3://sdcBucket/BLIP-main', url_or_filename))) as f:
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

    if not load_from_different_size:
        msg = model.load_state_dict(state_dict,strict=False)
        print('load checkpoint from %s'%url_or_filename)  
        return model,msg


    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % url_or_filename)
    return (model, msg)

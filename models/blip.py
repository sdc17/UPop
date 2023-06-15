import warnings
warnings.filterwarnings("ignore")

from models.vit import VisionTransformer, interpolate_pos_embed
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file

import torchvision
import math
import os
import io
from functools import reduce 

class BLIP_Base(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                 
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)  

        
    def forward(self, image, caption, mode):
        
        assert mode in ['image', 'text', 'multimodal'], "mode parameter must be image, text, or multimodal"
        text = self.tokenizer(caption, return_tensors="pt").to(image.device) 
        
        if mode=='image':    
            # return image features
            image_embeds = self.visual_encoder(image)             
            return image_embeds
        
        elif mode=='text':
            # return text features
            text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')  
            return text_output.last_hidden_state
        
        elif mode=='multimodal':
            # return multimodel features
            image_embeds = self.visual_encoder(image)    
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)      
            
            text.input_ids[:,0] = self.tokenizer.enc_token_id
            output = self.text_encoder(text.input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      )              
            return output.last_hidden_state
        
        
        
class BLIP_Decoder(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 prompt = 'a picture of ',
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
        self.layers = 12
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer, 0, search=search, evaluate=evaluate)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        med_config.search = search
        med_config.evaluate = evaluate
        self.text_decoder = BertLMHeadModel(config=med_config)    
        
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1
        
    def forward(self, image, caption):
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device) 
        
        text.input_ids[:,0] = self.tokenizer.bos_token_id
        
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)         
        decoder_targets[:,:self.prompt_length] = -100
     
        decoder_output = self.text_decoder(text.input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,                  
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )   
        loss_lm = decoder_output.loss
        
        return loss_lm
        
    def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image)

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
            
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}
        
        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device) 
        input_ids[:,0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1] 

        if sample:
            #nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  do_sample=True,
                                                  top_p=top_p,
                                                  num_return_sequences=1,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id, 
                                                  repetition_penalty=1.1,                                            
                                                  **model_kwargs)
        else:
            #beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  num_beams=num_beams,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,     
                                                  repetition_penalty=repetition_penalty,
                                                  **model_kwargs)            
            
        captions = []    
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)    
            captions.append(caption[len(self.prompt):])
        return captions
    
    
    def get_sparsity_loss(self):
        sparsity_loss_attn, sparsity_loss_mlp = 0, 0
        for i in range(self.layers):
            sparsity_loss_attn += torch.sum(torch.abs(getattr(self.visual_encoder.blocks, str(i)).attn.alpha))
            sparsity_loss_mlp += torch.sum(torch.abs(getattr(self.visual_encoder.blocks, str(i)).mlp.alpha))
            sparsity_loss_attn += torch.sum(torch.abs(getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.alpha))
            sparsity_loss_attn += torch.sum(torch.abs(getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.alpha))
            sparsity_loss_mlp += torch.sum(torch.abs(getattr(self.text_decoder.bert.encoder.layer, str(i)).intermediate.alpha))
        return sparsity_loss_attn, sparsity_loss_mlp


    def print_compression_statistics(self):
        mask_attn_vision_list, mask_attn_language_list  = [], []
        mask_mlp_vision_list, mask_mlp_language_list = [], []
        mask_cross_attn_list =  []
        reserved_ratio = lambda x: (torch.count_nonzero(x) / torch.numel(x)).item()
        for i in range(self.layers):
            mask_attn_vision_list.append(getattr(self.visual_encoder.blocks, str(i)).attn.alpha.data.view(-1))
            mask_mlp_vision_list.append(getattr(self.visual_encoder.blocks, str(i)).mlp.alpha.data.view(-1))
            mask_attn_language_list.append(getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.alpha.data.view(-1))
            mask_mlp_language_list.append(getattr(self.text_decoder.bert.encoder.layer, str(i)).intermediate.alpha.data.view(-1))
            mask_cross_attn_list.append(getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.alpha.data.view(-1))
        print_format = lambda x: [round(i * 100, 2) for i in x]
        print('mask_attn_vision:  ', print_format([reserved_ratio(x) for x in mask_attn_vision_list]))
        print('mask_attn_language: ', print_format([reserved_ratio(x) for x in mask_attn_language_list]))
        print('mask_cross_attn: ', print_format([reserved_ratio(x) for x in mask_cross_attn_list]))
        print('mask_mlp_vision: ', print_format([reserved_ratio(x) for x in mask_mlp_vision_list]))
        print('mask_mlp_language: ', print_format([reserved_ratio(x) for x in mask_mlp_language_list]))
        print('mask_vision: ', reserved_ratio(torch.cat(mask_attn_vision_list + mask_mlp_vision_list)))
        print('mask_language: ', reserved_ratio(torch.cat(mask_attn_language_list + mask_mlp_language_list)))
        print('mask_cross_attn: ', reserved_ratio(torch.cat(mask_cross_attn_list)))
        print('mask_attn: ', reserved_ratio(torch.cat(mask_attn_vision_list + mask_attn_language_list + mask_cross_attn_list)))
        print('mask_mlp: ', reserved_ratio(torch.cat(mask_mlp_vision_list + mask_mlp_language_list)))
        

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
            in_features = getattr(self.text_decoder.bert.encoder.layer, str(i)).intermediate.dense.weight.shape[-1]
            out_features = getattr(self.text_decoder.bert.encoder.layer, str(i)).output.dense.weight.shape[0]
            alpha = torch.squeeze(getattr(search_model.text_decoder.bert.encoder.layer, str(i)).intermediate.alpha.data)
            hidden_features = torch.count_nonzero(alpha)
            getattr(self.text_decoder.bert.encoder.layer, str(i)).intermediate.dense = nn.Linear(in_features, hidden_features)
            getattr(self.text_decoder.bert.encoder.layer, str(i)).intermediate.dense.weight.data = \
                getattr(search_model.text_decoder.bert.encoder.layer, str(i)).intermediate.dense.weight.data[alpha==1,:]
            getattr(self.text_decoder.bert.encoder.layer, str(i)).intermediate.dense.bias.data = \
                getattr(search_model.text_decoder.bert.encoder.layer, str(i)).intermediate.dense.bias.data[alpha==1]
            getattr(self.text_decoder.bert.encoder.layer, str(i)).output.dense = nn.Linear(hidden_features, out_features)
            getattr(self.text_decoder.bert.encoder.layer, str(i)).output.dense.weight.data = \
                getattr(search_model.text_decoder.bert.encoder.layer, str(i)).output.dense.weight.data[:, alpha==1]
            getattr(self.text_decoder.bert.encoder.layer, str(i)).output.dense.bias.data = \
                getattr(search_model.text_decoder.bert.encoder.layer, str(i)).output.dense.bias.data

            # bert attn
            in_features_query = getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.query.weight.shape[-1]
            in_features_key = getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.key.weight.shape[-1]
            in_features_value = getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.value.weight.shape[-1]
            out_features = getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.output.dense.weight.shape[0]
            alpha = torch.squeeze(getattr(search_model.text_decoder.bert.encoder.layer, str(i)).attention.self.alpha.data)
            hidden_features = torch.count_nonzero(alpha)
            getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.attention_head_size = hidden_features
            getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.all_head_size = \
                hidden_features * getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.num_attention_heads
            parameter_ratio = getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.num_attention_heads
            getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.query = nn.Linear(in_features_query, hidden_features)
            getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.query.weight.data = \
                getattr(search_model.text_decoder.bert.encoder.layer, str(i)).attention.self.query.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.query.bias.data = \
                getattr(search_model.text_decoder.bert.encoder.layer, str(i)).attention.self.query.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.key = nn.Linear(in_features_key, hidden_features)
            getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.key.weight.data = \
                getattr(search_model.text_decoder.bert.encoder.layer, str(i)).attention.self.key.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.key.bias.data = \
                getattr(search_model.text_decoder.bert.encoder.layer, str(i)).attention.self.key.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.value = nn.Linear(in_features_value, hidden_features)
            getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.value.weight.data = \
                getattr(search_model.text_decoder.bert.encoder.layer, str(i)).attention.self.value.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.value.bias.data = \
                getattr(search_model.text_decoder.bert.encoder.layer, str(i)).attention.self.value.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.output.dense = nn.Linear(hidden_features, out_features)
            getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.output.dense.weight.data = \
                getattr(search_model.text_decoder.bert.encoder.layer, str(i)).attention.output.dense.weight.data[:, alpha.repeat(parameter_ratio)==1]
            getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.output.dense.bias.data = \
                getattr(search_model.text_decoder.bert.encoder.layer, str(i)).attention.output.dense.bias.data
        
            # corss att
            in_features_query = getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.query.weight.shape[-1]
            in_features_key = getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.key.weight.shape[-1]
            in_features_value = getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.value.weight.shape[-1]
            out_features = getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.output.dense.weight.shape[0]
            alpha = torch.squeeze(getattr(search_model.text_decoder.bert.encoder.layer, str(i)).crossattention.self.alpha.data)
            hidden_features = torch.count_nonzero(alpha)
            getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.attention_head_size = hidden_features
            getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.all_head_size = \
                hidden_features * getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.num_attention_heads
            parameter_ratio = getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.num_attention_heads
            getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.query = nn.Linear(in_features_query, hidden_features)
            getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.query.weight.data = \
                getattr(search_model.text_decoder.bert.encoder.layer, str(i)).crossattention.self.query.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.query.bias.data = \
                getattr(search_model.text_decoder.bert.encoder.layer, str(i)).crossattention.self.query.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.key = nn.Linear(in_features_key, hidden_features)
            getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.key.weight.data = \
                getattr(search_model.text_decoder.bert.encoder.layer, str(i)).crossattention.self.key.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.key.bias.data = \
                getattr(search_model.text_decoder.bert.encoder.layer, str(i)).crossattention.self.key.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.value = nn.Linear(in_features_value, hidden_features)
            getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.value.weight.data = \
                getattr(search_model.text_decoder.bert.encoder.layer, str(i)).crossattention.self.value.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.value.bias.data = \
                getattr(search_model.text_decoder.bert.encoder.layer, str(i)).crossattention.self.value.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.output.dense = nn.Linear(hidden_features, out_features)
            getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.output.dense.weight.data = \
                getattr(search_model.text_decoder.bert.encoder.layer, str(i)).crossattention.output.dense.weight.data[:, alpha.repeat(parameter_ratio)==1]
            getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.output.dense.bias.data = \
                getattr(search_model.text_decoder.bert.encoder.layer, str(i)).crossattention.output.dense.bias.data

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
            if getattr(self.visual_encoder.blocks, str(i)).mlp.fc1.weight.shape != state_dict['visual_encoder.blocks.' + str(i) + '.mlp.fc1.weight'].shape:
                del getattr(self.visual_encoder.blocks, str(i)).mlp.fc1
                getattr(self.visual_encoder.blocks, str(i)).mlp.fc1 = nn.Linear(*state_dict['visual_encoder.blocks.' + str(i) + '.mlp.fc1.weight'].shape[::-1])
                del getattr(self.visual_encoder.blocks, str(i)).mlp.fc2
                getattr(self.visual_encoder.blocks, str(i)).mlp.fc2 = nn.Linear(*state_dict['visual_encoder.blocks.' + str(i) + '.mlp.fc2.weight'].shape[::-1])

            # vit attn
            if getattr(self.visual_encoder.blocks, str(i)).attn.qkv.weight.shape != state_dict['visual_encoder.blocks.' + str(i) + '.attn.qkv.weight'].shape:
                del getattr(self.visual_encoder.blocks, str(i)).attn.qkv
                getattr(self.visual_encoder.blocks, str(i)).attn.qkv = nn.Linear(*state_dict['visual_encoder.blocks.' + str(i) + '.attn.qkv.weight'].shape[::-1])
                del getattr(self.visual_encoder.blocks, str(i)).attn.proj
                getattr(self.visual_encoder.blocks, str(i)).attn.proj = nn.Linear(*state_dict['visual_encoder.blocks.' + str(i) + '.attn.proj.weight'].shape[::-1])
            
            # bert mlp
            if getattr(self.text_decoder.bert.encoder.layer, str(i)).intermediate.dense.weight.shape != state_dict['text_decoder.bert.encoder.layer.' + str(i) + '.intermediate.dense.weight'].shape:
                del getattr(self.text_decoder.bert.encoder.layer, str(i)).intermediate.dense
                getattr(self.text_decoder.bert.encoder.layer, str(i)).intermediate.dense = \
                    nn.Linear(*state_dict['text_decoder.bert.encoder.layer.' + str(i) + '.intermediate.dense.weight'].shape[::-1])
                del getattr(self.text_decoder.bert.encoder.layer, str(i)).output.dense
                getattr(self.text_decoder.bert.encoder.layer, str(i)).output.dense = \
                    nn.Linear(*state_dict['text_decoder.bert.encoder.layer.' + str(i) + '.output.dense.weight'].shape[::-1])

            # bert attn
            if getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.query.weight.shape != state_dict['text_decoder.bert.encoder.layer.'+str(i)+'.attention.self.query.weight'].shape:
                getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.attention_head_size = state_dict['text_decoder.bert.encoder.layer.'+str(i)+'.attention.self.query.weight'].shape[-2] // \
                    getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.num_attention_heads
                getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.all_head_size = getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.attention_head_size * \
                    getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.num_attention_heads
                del getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.query
                getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.query = nn.Linear(*state_dict['text_decoder.bert.encoder.layer.'+str(i)+'.attention.self.query.weight'].shape[::-1])
                del getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.key
                getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.key = nn.Linear(*state_dict['text_decoder.bert.encoder.layer.'+str(i)+'.attention.self.key.weight'].shape[::-1])
                del getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.value
                getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.value = nn.Linear(*state_dict['text_decoder.bert.encoder.layer.'+str(i)+'.attention.self.value.weight'].shape[::-1])
                del getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.output.dense
                getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.output.dense = nn.Linear(*state_dict['text_decoder.bert.encoder.layer.'+str(i)+'.attention.output.dense.weight'].shape[::-1])
        
            # corss att
            if getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.query.weight.shape != state_dict['text_decoder.bert.encoder.layer.'+str(i)+'.crossattention.self.query.weight'].shape:
                getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.attention_head_size = state_dict['text_decoder.bert.encoder.layer.'+str(i)+'.crossattention.self.query.weight'].shape[-2] // \
                    getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.num_attention_heads
                getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.all_head_size = getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.attention_head_size * \
                    getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.num_attention_heads
                del getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.query
                getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.query = nn.Linear(*state_dict['text_decoder.bert.encoder.layer.'+str(i)+'.crossattention.self.query.weight'].shape[::-1])
                del getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.key
                getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.key = nn.Linear(*state_dict['text_decoder.bert.encoder.layer.'+str(i)+'.crossattention.self.key.weight'].shape[::-1])
                del getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.value
                getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.value = nn.Linear(*state_dict['text_decoder.bert.encoder.layer.'+str(i)+'.crossattention.self.value.weight'].shape[::-1])
                del getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.output.dense
                getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.output.dense = nn.Linear(*state_dict['text_decoder.bert.encoder.layer.'+str(i)+'.crossattention.output.dense.weight'].shape[::-1])

        # torch.cuda.empty_cache()
        self.load_state_dict(state_dict, strict=False)

def blip_decoder(client, pretrained='',**kwargs):
    model = BLIP_Decoder(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained, client)
        # assert(len(msg.missing_keys)==0)
        print("missing keys:")
        print(msg.missing_keys)
    return model    
    
def blip_feature_extractor(pretrained='',**kwargs):
    model = BLIP_Base(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model        

def init_tokenizer():
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('./pretrained/bert-base-uncased/')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer


def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0, search=False, evaluate=False):
    if vit=='base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12, 
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate,
                                           search=search, evaluate=evaluate
                                          )   
    elif vit=='large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24, 
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate,
                                           search=search, evaluate=evaluate
                                          )   
    return visual_encoder, vision_width

def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def load_checkpoint(model,url_or_filename, client=None):
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
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)    
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                del state_dict[key]
    msg = model.load_state_dict(state_dict,strict=False)
    print('load checkpoint from %s'%url_or_filename)  
    return model,msg
    
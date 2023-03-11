from tkinter.messagebox import NO
from models.med import BertConfig, BertModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

from models.blip import create_vit, init_tokenizer, load_checkpoint
from functools import reduce 

class BLIP_Retrieval(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                      
                 embed_dim = 256,     
                 queue_size = 57600,
                 momentum = 0.995,
                 negative_all_rank = False,
                 embeddings = None,
                 layers = None,
                 search=False,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        self.layers = 12
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer, 0, embeddings, layers, search=search)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        med_config.search = search
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)          

        text_width = self.text_encoder.config.hidden_size
        
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2) 
        
        # create momentum encoders  
        self.visual_encoder_m, vision_width = create_vit(vit,image_size, False, 0, 0, embeddings, layers)    
        # self.visual_encoder_m, vision_width = create_vit(vit,image_size, False, 0, 0, 768, 12)           
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        # self.text_encoder_m = BertModel(config=med_config, add_pooling_layer=False)    
        med_config_m = med_config
        med_config_m.search = False
        self.text_encoder_m = BertModel(config=med_config_m, add_pooling_layer=False)   
        self.text_proj_m = nn.Linear(text_width, embed_dim)

        # # create momentum encoders  
        # self.visual_encoder_m, vision_width = create_vit(vit,image_size, False, 0, 0, embeddings, layers, search=search)        
        # self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        # self.text_encoder_m = BertModel(config=med_config, add_pooling_layer=False)    
        # self.text_proj_m = nn.Linear(text_width, embed_dim)

        
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]       
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("idx_queue", torch.full((1,queue_size),-100))
        self.register_buffer("ptr_queue", torch.zeros(1, dtype=torch.long))  

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        
        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07*torch.ones([]))   
        
        self.negative_all_rank = negative_all_rank
        
        
    def forward(self, image, caption, alpha, idx):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)    
        
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35, 
                              return_tensors="pt").to(image.device) 
        
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)        

        ###============== Image-text Contrastive Learning ===================###
        idx = idx.view(-1,1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()],dim=1)  
        pos_idx = torch.eq(idx, idx_all).float()       
        sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)   
        
        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image) 
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  
            image_feat_m_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)                   
            
            text_output_m = self.text_encoder_m(text.input_ids, attention_mask = text.attention_mask,                      
                                                return_dict = True, mode = 'text')    
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
            text_feat_m_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            sim_i2t_m = image_feat_m @ text_feat_m_all / self.temp  
            sim_t2i_m = text_feat_m @ image_feat_m_all / self.temp 

            # sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            # sim_targets.fill_diagonal_(1)          

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

        sim_i2t = image_feat @ text_feat_m_all / self.temp 
        sim_t2i = text_feat @ image_feat_m_all / self.temp 
                             
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

        loss_ita = (loss_i2t+loss_t2i)/2
        
        idxs = concat_all_gather(idx)
        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idxs)        

        ###============== Image-text Matching ===================###
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:,0] = self.tokenizer.enc_token_id

        # forward the positve image-text pair
        bs = image.size(0)
        output_pos = self.text_encoder(encoder_input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      )  
        
        
        if self.negative_all_rank:    
            # compute sample similarity
            with torch.no_grad():                
                mask = torch.eq(idx, idxs.t())

                image_feat_world = concat_all_gather(image_feat)
                text_feat_world = concat_all_gather(text_feat)

                sim_i2t = image_feat @ text_feat_world.t() / self.temp 
                sim_t2i = text_feat @ image_feat_world.t() / self.temp 

                weights_i2t = F.softmax(sim_i2t,dim=1)
                weights_i2t.masked_fill_(mask, 0)            

                weights_t2i = F.softmax(sim_t2i,dim=1)
                weights_t2i.masked_fill_(mask, 0)     

            image_embeds_world = all_gather_with_grad(image_embeds) 

            # select a negative image (from all ranks) for each text
            image_embeds_neg = []    
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds_world[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

            # select a negative text (from all ranks) for each image
            input_ids_world = concat_all_gather(encoder_input_ids)
            att_mask_world = concat_all_gather(text.attention_mask)        

            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(input_ids_world[neg_idx])
                text_atts_neg.append(att_mask_world[neg_idx])
                
        else:
            with torch.no_grad():                
                mask = torch.eq(idx, idx.t())
                
                sim_i2t = image_feat @ text_feat.t() / self.temp 
                sim_t2i = text_feat @ image_feat.t() / self.temp 

                weights_i2t = F.softmax(sim_i2t,dim=1)
                weights_i2t.masked_fill_(mask, 0)            

                weights_t2i = F.softmax(sim_t2i,dim=1)
                weights_t2i.masked_fill_(mask, 0)     

            # select a negative image (from same rank) for each text
            image_embeds_neg = []    
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

            # select a negative text (from same rank) for each image    
            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(encoder_input_ids[neg_idx])
                text_atts_neg.append(text.attention_mask[neg_idx])            
            
        text_ids_neg = torch.stack(text_ids_neg,dim=0)   
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg],dim=0)     
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0)     

        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        output_neg = self.text_encoder(text_ids_all,
                                       attention_mask = text_atts_all,
                                       encoder_hidden_states = image_embeds_all,
                                       encoder_attention_mask = image_atts_all,      
                                       return_dict = True,
                                      )                         
          

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)            

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)     

        return loss_ita, loss_itm 


    @torch.no_grad()    
    def copy_params(self):
        remove_alpha = lambda model: [param for name, param in list(model.named_parameters()) if not ('alpha' in name)]
        for model_pair in self.model_pairs:           
            for param, param_m in zip(remove_alpha(model_pair[0]), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        remove_alpha = lambda model: [param for name, param in list(model.named_parameters()) if not ('alpha' in name)]
        for model_pair in self.model_pairs:           
            for param, param_m in zip(remove_alpha(model_pair[0]), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idxs):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        

        batch_size = image_feats.shape[0]

        ptr = int(self.ptr_queue)
        assert self.queue_size % batch_size == 0  # for simplicity

        if ptr % batch_size != 0:
            ptr = (ptr // batch_size) * batch_size

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size # move pointer

        self.ptr_queue[0] = ptr  

        
    def get_sparsity_loss(self):
        sparsity_loss_attn, sparsity_loss_mlp = 0, 0
        for i in range(self.layers):
            sparsity_loss_attn += torch.sum(torch.abs(getattr(self.visual_encoder.blocks, str(i)).attn.alpha))
            sparsity_loss_mlp += torch.sum(torch.abs(getattr(self.visual_encoder.blocks, str(i)).mlp.alpha))
            sparsity_loss_attn += torch.sum(torch.abs(getattr(self.text_encoder.encoder.layer, str(i)).attention.self.alpha))
            sparsity_loss_attn += torch.sum(torch.abs(getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self.alpha))
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

            getattr(self.visual_encoder_m.blocks, str(i)).mlp.fc1 = nn.Linear(in_features, hidden_features)
            getattr(self.visual_encoder_m.blocks, str(i)).mlp.fc1.weight.data = getattr(search_model.visual_encoder.blocks, str(i)).mlp.fc1.weight.data[alpha==1,:]
            getattr(self.visual_encoder_m.blocks, str(i)).mlp.fc1.bias.data = getattr(search_model.visual_encoder.blocks, str(i)).mlp.fc1.bias.data[alpha==1]
            getattr(self.visual_encoder_m.blocks, str(i)).mlp.fc2 = nn.Linear(hidden_features, out_features)
            getattr(self.visual_encoder_m.blocks, str(i)).mlp.fc2.weight.data = getattr(search_model.visual_encoder.blocks, str(i)).mlp.fc2.weight.data[:, alpha==1]
            getattr(self.visual_encoder_m.blocks, str(i)).mlp.fc2.bias.data = getattr(search_model.visual_encoder.blocks, str(i)).mlp.fc2.bias.data

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

            getattr(self.visual_encoder_m.blocks, str(i)).attn.qkv = nn.Linear(in_features, hidden_features*parameter_ratio)
            getattr(self.visual_encoder_m.blocks, str(i)).attn.qkv.weight.data = \
                getattr(search_model.visual_encoder.blocks, str(i)).attn.qkv.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.visual_encoder_m.blocks, str(i)).attn.qkv.weight.bias = \
                getattr(search_model.visual_encoder.blocks, str(i)).attn.qkv.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.visual_encoder_m.blocks, str(i)).attn.proj = nn.Linear(hidden_features, out_features)
            getattr(self.visual_encoder_m.blocks, str(i)).attn.proj.weight.data = \
                getattr(search_model.visual_encoder.blocks, str(i)).attn.proj.weight.data[:, alpha.repeat(parameter_ratio//3)==1]
            getattr(self.visual_encoder_m.blocks, str(i)).attn.proj.bias.data = getattr(search_model.visual_encoder.blocks, str(i)).attn.proj.bias.data

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
            
            getattr(self.text_encoder_m.encoder.layer, str(i)).intermediate.dense = nn.Linear(in_features, hidden_features)
            getattr(self.text_encoder_m.encoder.layer, str(i)).intermediate.dense.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).intermediate.dense.weight.data[alpha==1,:]
            getattr(self.text_encoder_m.encoder.layer, str(i)).intermediate.dense.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).intermediate.dense.bias.data[alpha==1]
            getattr(self.text_encoder_m.encoder.layer, str(i)).output.dense = nn.Linear(hidden_features, out_features)
            getattr(self.text_encoder_m.encoder.layer, str(i)).output.dense.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).output.dense.weight.data[:, alpha==1]
            getattr(self.text_encoder_m.encoder.layer, str(i)).output.dense.bias.data = \
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

            getattr(self.text_encoder_m.encoder.layer, str(i)).attention.self.attention_head_size = hidden_features
            getattr(self.text_encoder_m.encoder.layer, str(i)).attention.self.all_head_size = \
                hidden_features * getattr(self.text_encoder_m.encoder.layer, str(i)).attention.self.num_attention_heads
            parameter_ratio = getattr(self.text_encoder_m.encoder.layer, str(i)).attention.self.num_attention_heads
            getattr(self.text_encoder_m.encoder.layer, str(i)).attention.self.query = nn.Linear(in_features_query, hidden_features)
            getattr(self.text_encoder_m.encoder.layer, str(i)).attention.self.query.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).attention.self.query.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.text_encoder_m.encoder.layer, str(i)).attention.self.query.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).attention.self.query.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.text_encoder_m.encoder.layer, str(i)).attention.self.key = nn.Linear(in_features_key, hidden_features)
            getattr(self.text_encoder_m.encoder.layer, str(i)).attention.self.key.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).attention.self.key.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.text_encoder_m.encoder.layer, str(i)).attention.self.key.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).attention.self.key.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.text_encoder_m.encoder.layer, str(i)).attention.self.value = nn.Linear(in_features_value, hidden_features)
            getattr(self.text_encoder_m.encoder.layer, str(i)).attention.self.value.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).attention.self.value.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.text_encoder_m.encoder.layer, str(i)).attention.self.value.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).attention.self.value.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.text_encoder_m.encoder.layer, str(i)).attention.output.dense = nn.Linear(hidden_features, out_features)
            getattr(self.text_encoder_m.encoder.layer, str(i)).attention.output.dense.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).attention.output.dense.weight.data[:, alpha.repeat(parameter_ratio)==1]
            getattr(self.text_encoder_m.encoder.layer, str(i)).attention.output.dense.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).attention.output.dense.bias.data
        
            # corss att
            in_features_query = getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self.query.weight.shape[-1]
            in_features_key = getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self.key.weight.shape[-1]
            in_features_value = getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self.value.weight.shape[-1]
            out_features = getattr(self.text_encoder.encoder.layer, str(i)).crossattention.output.dense.weight.shape[0]
            alpha = torch.squeeze(getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self.alpha.data)
            hidden_features = torch.count_nonzero(alpha)
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self.attention_head_size = hidden_features
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self.all_head_size = \
                hidden_features * getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self.num_attention_heads
            parameter_ratio = getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self.num_attention_heads
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self.query = nn.Linear(in_features_query, hidden_features)
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self.query.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self.query.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self.query.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self.query.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self.key = nn.Linear(in_features_key, hidden_features)
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self.key.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self.key.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self.key.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self.key.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self.value = nn.Linear(in_features_value, hidden_features)
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self.value.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self.value.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self.value.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self.value.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.output.dense = nn.Linear(hidden_features, out_features)
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.output.dense.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.output.dense.weight.data[:, alpha.repeat(parameter_ratio)==1]
            getattr(self.text_encoder.encoder.layer, str(i)).crossattention.output.dense.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.output.dense.bias.data

            getattr(self.text_encoder_m.encoder.layer, str(i)).crossattention.self.attention_head_size = hidden_features
            getattr(self.text_encoder_m.encoder.layer, str(i)).crossattention.self.all_head_size = \
                hidden_features * getattr(self.text_encoder_m.encoder.layer, str(i)).crossattention.self.num_attention_heads
            parameter_ratio = getattr(self.text_encoder_m.encoder.layer, str(i)).crossattention.self.num_attention_heads
            getattr(self.text_encoder_m.encoder.layer, str(i)).crossattention.self.query = nn.Linear(in_features_query, hidden_features)
            getattr(self.text_encoder_m.encoder.layer, str(i)).crossattention.self.query.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self.query.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.text_encoder_m.encoder.layer, str(i)).crossattention.self.query.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self.query.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.text_encoder_m.encoder.layer, str(i)).crossattention.self.key = nn.Linear(in_features_key, hidden_features)
            getattr(self.text_encoder_m.encoder.layer, str(i)).crossattention.self.key.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self.key.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.text_encoder_m.encoder.layer, str(i)).crossattention.self.key.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self.key.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.text_encoder_m.encoder.layer, str(i)).crossattention.self.value = nn.Linear(in_features_value, hidden_features)
            getattr(self.text_encoder_m.encoder.layer, str(i)).crossattention.self.value.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self.value.weight.data[alpha.repeat(parameter_ratio)==1,:]
            getattr(self.text_encoder_m.encoder.layer, str(i)).crossattention.self.value.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.self.value.bias.data[alpha.repeat(parameter_ratio)==1]
            getattr(self.text_encoder_m.encoder.layer, str(i)).crossattention.output.dense = nn.Linear(hidden_features, out_features)
            getattr(self.text_encoder_m.encoder.layer, str(i)).crossattention.output.dense.weight.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.output.dense.weight.data[:, alpha.repeat(parameter_ratio)==1]
            getattr(self.text_encoder_m.encoder.layer, str(i)).crossattention.output.dense.bias.data = \
                getattr(search_model.text_encoder.encoder.layer, str(i)).crossattention.output.dense.bias.data


    def print_compression_statistics(self):
        mask_attn_vision_list, mask_attn_language_list  = [], []
        mask_mlp_vision_list, mask_mlp_language_list = [], []
        mask_cross_attn_list = []
        reserved_ratio = lambda x: (torch.count_nonzero(x) / torch.numel(x)).item()
        for i in range(self.layers):
            mask_attn_vision_list.append(getattr(self.visual_encoder.blocks, str(i)).attn.alpha.data.view(-1))
            mask_mlp_vision_list.append(getattr(self.visual_encoder.blocks, str(i)).mlp.alpha.data.view(-1))
            mask_attn_language_list.append(getattr(self.text_encoder.encoder.layer, str(i)).attention.self.alpha.data.view(-1))
            mask_mlp_language_list.append(getattr(self.text_encoder.encoder.layer, str(i)).intermediate.alpha.data.view(-1))
            mask_cross_attn_list.append(getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self.alpha.data.view(-1))
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


def blip_retrieval(client, pretrained='',**kwargs):
    model = BLIP_Retrieval(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained,kwargs['vit']=='custom', client)
        print("missing keys:")
        print(msg.missing_keys)
    return model 


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output      


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)

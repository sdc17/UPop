from models.med import BertConfig, BertModel, BertLMHeadModel
from models.blip import create_vit, init_tokenizer, load_checkpoint

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np

class BLIP_VQA(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 480,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
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

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1, search=search)
        self.tokenizer = init_tokenizer()  
        
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        encoder_config.search = search
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False) 
        
        decoder_config = BertConfig.from_json_file(med_config)    
        decoder_config.search = search
        self.text_decoder = BertLMHeadModel(config=decoder_config)          


    def forward(self, image, question, answer=None, n=None, weights=None, train=True, inference='rank', k_test=128):
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        question = self.tokenizer(question, padding='longest', truncation=True, max_length=35, 
                                  return_tensors="pt").to(image.device) 
        question.input_ids[:,0] = self.tokenizer.enc_token_id
        
        if train:               
            '''
            n: number of answers for each question
            weights: weight for each answer
            '''                     
            answer = self.tokenizer(answer, padding='longest', return_tensors="pt").to(image.device) 
            answer.input_ids[:,0] = self.tokenizer.bos_token_id
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)      

            question_output = self.text_encoder(question.input_ids, 
                                                attention_mask = question.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                             
                                                return_dict = True)    

            question_states = []                
            question_atts = []  
            for b, n in enumerate(n):
                question_states += [question_output.last_hidden_state[b]]*n
                question_atts += [question.attention_mask[b]]*n                
            question_states = torch.stack(question_states,0)    
            question_atts = torch.stack(question_atts,0)     

            answer_output = self.text_decoder(answer.input_ids, 
                                              attention_mask = answer.attention_mask, 
                                              encoder_hidden_states = question_states,
                                              encoder_attention_mask = question_atts,                  
                                              labels = answer_targets,
                                              return_dict = True,   
                                              reduction = 'none',
                                             )      
            
            loss = weights * answer_output.loss
            loss = loss.sum()/image.size(0)

            return loss
            

        else: 
            question_output = self.text_encoder(question.input_ids, 
                                                attention_mask = question.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                                    
                                                return_dict = True) 
            
            if inference=='generate':
                num_beams = 3
                question_states = question_output.last_hidden_state.repeat_interleave(num_beams,dim=0)
                question_atts = torch.ones(question_states.size()[:-1],dtype=torch.long).to(question_states.device)
                model_kwargs = {"encoder_hidden_states": question_states, "encoder_attention_mask":question_atts}
                
                bos_ids = torch.full((image.size(0),1),fill_value=self.tokenizer.bos_token_id,device=image.device)
                
                outputs = self.text_decoder.generate(input_ids=bos_ids,
                                                     max_length=10,
                                                     min_length=1,
                                                     num_beams=num_beams,
                                                     eos_token_id=self.tokenizer.sep_token_id,
                                                     pad_token_id=self.tokenizer.pad_token_id, 
                                                     **model_kwargs)
                
                answers = []    
                for output in outputs:
                    answer = self.tokenizer.decode(output, skip_special_tokens=True)    
                    answers.append(answer)
                return answers
            
            elif inference=='rank':
                max_ids = self.rank_answer(question_output.last_hidden_state, question.attention_mask, 
                                           answer.input_ids, answer.attention_mask, k_test) 
                return max_ids
 
                
                
    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        
        num_ques = question_states.size(0)
        start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token
        
        start_output = self.text_decoder(start_ids, 
                                         encoder_hidden_states = question_states,
                                         encoder_attention_mask = question_atts,                                      
                                         return_dict = True,
                                         reduction = 'none')              
        logits = start_output.logits[:,0,:] # first token's logit
        
        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:,1]
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) 
        topk_probs, topk_ids = prob_first_token.topk(k,dim=1) 
        
        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids,dim=0)  
        input_atts = torch.cat(input_atts,dim=0)  

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)
        
        output = self.text_decoder(input_ids, 
                                   attention_mask = input_atts, 
                                   encoder_hidden_states = question_states,
                                   encoder_attention_mask = question_atts,     
                                   labels = targets_ids,
                                   return_dict = True, 
                                   reduction = 'none')   
        
        log_probs_sum = -output.loss
        log_probs_sum = log_probs_sum.view(num_ques,k)

        max_topk_ids = log_probs_sum.argmax(dim=1) 
        max_ids = topk_ids[max_topk_ids>=0,max_topk_ids]

        return max_ids

    def get_sparsity_loss(self):
        sparsity_loss_attn, sparsity_loss_mlp = 0, 0
        for i in range(self.layers):
            sparsity_loss_attn += torch.sum(torch.abs(getattr(self.visual_encoder.blocks, str(i)).attn.alpha))
            sparsity_loss_mlp += torch.sum(torch.abs(getattr(self.visual_encoder.blocks, str(i)).mlp.alpha))
            sparsity_loss_attn += torch.sum(torch.abs(getattr(self.text_encoder.encoder.layer, str(i)).attention.self.alpha))
            sparsity_loss_attn += torch.sum(torch.abs(getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self.alpha))
            sparsity_loss_mlp += torch.sum(torch.abs(getattr(self.text_encoder.encoder.layer, str(i)).intermediate.alpha))
            sparsity_loss_attn += torch.sum(torch.abs(getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.alpha))
            sparsity_loss_attn += torch.sum(torch.abs(getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.alpha))
            sparsity_loss_mlp += torch.sum(torch.abs(getattr(self.text_decoder.bert.encoder.layer, str(i)).intermediate.alpha))
        return sparsity_loss_attn, sparsity_loss_mlp

    def print_compression_statistics(self):
        mask_attn_vision_list, mask_attn_language_encoder_list, mask_attn_language_decoder_list  = [], [], []
        mask_cross_attn_encoder_list, mask_cross_attn_decoder_list = [], []
        mask_mlp_vision_list, mask_mlp_language_encoder_list, mask_mlp_language_decoder_list = [], [], []
        reserved_ratio = lambda x: (torch.count_nonzero(x) / torch.numel(x)).item()
        for i in range(self.layers):
            mask_attn_vision_list.append(getattr(self.visual_encoder.blocks, str(i)).attn.alpha.data.view(-1))
            mask_mlp_vision_list.append(getattr(self.visual_encoder.blocks, str(i)).mlp.alpha.data.view(-1))
            mask_attn_language_encoder_list.append(getattr(self.text_encoder.encoder.layer, str(i)).attention.self.alpha.data.view(-1))
            mask_attn_language_decoder_list.append(getattr(self.text_decoder.bert.encoder.layer, str(i)).attention.self.alpha.data.view(-1))
            mask_mlp_language_encoder_list.append(getattr(self.text_encoder.encoder.layer, str(i)).intermediate.alpha.data.view(-1))
            mask_mlp_language_decoder_list.append(getattr(self.text_decoder.bert.encoder.layer, str(i)).intermediate.alpha.data.view(-1))
            mask_cross_attn_encoder_list.append(getattr(self.text_encoder.encoder.layer, str(i)).crossattention.self.alpha.data.view(-1))
            mask_cross_attn_decoder_list.append(getattr(self.text_decoder.bert.encoder.layer, str(i)).crossattention.self.alpha.data.view(-1))
    
        print_format = lambda x: [round(i * 100, 2) for i in x]
        print('mask_attn_vision:  ', print_format([reserved_ratio(x) for x in mask_attn_vision_list]))
        print('mask_attn_language_encoder: ', print_format([reserved_ratio(x) for x in mask_attn_language_encoder_list]))
        print('mask_attn_language_decoder: ', print_format([reserved_ratio(x) for x in mask_attn_language_decoder_list]))
        print('mask_cross_attn_encoder: ', print_format([reserved_ratio(x) for x in mask_cross_attn_encoder_list]))
        print('mask_cross_attn_decoder: ', print_format([reserved_ratio(x) for x in mask_cross_attn_decoder_list]))
        print('mask_mlp_vision: ', print_format([reserved_ratio(x) for x in mask_mlp_vision_list]))
        print('mask_mlp_language_encoder: ', print_format([reserved_ratio(x) for x in mask_mlp_language_encoder_list]))
        print('mask_mlp_language_decoder: ', print_format([reserved_ratio(x) for x in mask_mlp_language_decoder_list]))
        print('mask_vision: ', reserved_ratio(torch.cat(mask_attn_vision_list + mask_mlp_vision_list)))
        print('mask_language: ', reserved_ratio(torch.cat(mask_attn_language_encoder_list + mask_attn_language_decoder_list + mask_mlp_language_encoder_list + mask_mlp_language_decoder_list)))
        print('mask_cross_attn: ', reserved_ratio(torch.cat(mask_cross_attn_encoder_list + mask_cross_attn_decoder_list)))
        print('mask_attn: ', reserved_ratio(torch.cat(mask_attn_vision_list + mask_attn_language_encoder_list + mask_attn_language_decoder_list + mask_cross_attn_encoder_list + mask_cross_attn_decoder_list)))
        print('mask_mlp: ', reserved_ratio(torch.cat(mask_mlp_vision_list + mask_mlp_language_encoder_list + mask_mlp_language_decoder_list)))

    
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

            # bert encoder mlp
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

            # bert decoder mlp
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

            # bert encoder attn
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

            # bert decoder attn
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

            # corss endcoder att 
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

            # corss decoder att 
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

            
def blip_vqa(client, pretrained='',**kwargs):
    model = BLIP_VQA(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained,False, client)
#         assert(len(msg.missing_keys)==0)
        print("missing keys:")
        print(msg.missing_keys)
    return model  


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))    
        
        
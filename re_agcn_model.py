import copy
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import torch.nn.functional as F
from model.bert import BertPreTrainedModel, BertModel
from model.agcn import TypeGraphConvolution

class ReAgcn(BertPreTrainedModel):
    def __init__(self, config):
        super(ReAgcn, self).__init__(config)
        self.bert = BertModel(config)
        self.dep_type_embedding = nn.Embedding(config.type_num, config.hidden_size, padding_idx=0)

        gcn_layer = TypeGraphConvolution(config.hidden_size, config.hidden_size)
        self.entity_hidden_size = config.entity_hidden_size
        first_gcn_layer = TypeGraphConvolution(config.hidden_size + config.entity_hidden_size, config.hidden_size)
        self.gcn_layer = nn.ModuleList([first_gcn_layer if _ == 0 else copy.deepcopy(gcn_layer) for _ in range(config.num_gcn_layers)])

        self.ensemble_linear = nn.Linear(1, config.num_gcn_layers)
        self.emb_dropout = nn.Dropout(config.emb_dropout_prob)
        self.gcn_dropout = nn.Dropout(config.gcn_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*3, config.num_labels)
        self.apply(self.init_bert_weights)
        
        #zhao_add
        linear_op = nn.Linear(config.hidden_size,config.hidden_size)
        first_linear_op = nn.Linear(config.hidden_size+config.entity_hidden_size, config.hidden_size+config.entity_hidden_size)
        self.linear_positive_op = nn.ModuleList([first_linear_op if _ == 0 else copy.deepcopy(linear_op) for _ in range(config.num_gcn_layers)])
        self.linear_reverse_op = nn.ModuleList([first_linear_op if _ == 0 else copy.deepcopy(linear_op) for _ in range(config.num_gcn_layers)])
        
        #add networks for entity-aware module
        self.We_linear = nn.Linear(config.hidden_size*2,config.entity_hidden_size)
        self.Wg_linear = nn.Linear(config.hidden_size + config.entity_hidden_size , config.hidden_size + config.entity_hidden_size)
        
    def valid_filter(self, sequence_output, valid_ids):
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=sequence_output.dtype,
                                   device=sequence_output.device)
        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            valid_output[i][:temp.size(0)] = temp
        return valid_output

    def max_pooling(self, sequence, e_mask):
        entity_output = sequence * torch.stack([e_mask] * sequence.shape[-1], 2) + torch.stack(
            [(1.0 - e_mask) * -1000.0] * sequence.shape[-1], 2)
        entity_output = torch.max(entity_output, -2)[0]
        return entity_output.type_as(sequence)

    def extract_entity(self, sequence, e_mask):
        return self.max_pooling(sequence, e_mask)

    def get_attention(self, val_out, dep_embed, adj, i):
        batch_size, max_len, feat_dim = val_out.shape
        
        val_us = val_out.unsqueeze(dim=2)
        val_us = val_us.repeat(1,1,max_len,1)
        
        val_reverse = val_us.reshape(batch_size*max_len*max_len,feat_dim)
        val_positive = val_us.view(batch_size*max_len*max_len,feat_dim) 

        val_positive = self.linear_positive_op[i](val_positive)
        val_reverse = self.linear_reverse_op[i](val_reverse)

        val_positive = val_positive.view(batch_size,max_len,max_len,feat_dim)
        val_reverse = val_reverse.view(batch_size,max_len,max_len,feat_dim)

        val_positive = torch.cat((val_positive,dep_embed),axis=-1)
        val_positive = torch.cat((val_positive,dep_embed),axis=-1)

        atten_expand_positive = (val_positive.float() * val_positive.float().transpose(1,2))
        atten_expand_reverse = (val_reverse.float() * val_reverse.float().transpose(1,2))

        atten_expand_positive = torch.sum(atten_expand_positive,dim=-1)
        atten_expand_positive = atten_expand_positive / feat_dim ** 0.5
        atten_expand_reverse = torch.sum(atten_expand_reverse,dim=-1)
        atten_expand_reverse = atten_expand_reverse / feat_dim ** 0.5

        adj_reverse = torch.clamp(adj,-1,0)
        adj_reverse = torch.abs(adj_reverse) 
        adj_positive = torch.clamp(adj,1)
               
        attention_score = atten_expand_positive * adj_positive + atten_expand_reverse * adj_reverse
        
        #softmax
        exp_attention_score = torch.exp(attention_score)
        sum_attention_score = torch.sum(exp_attention_score, dim=-1, keepdim = True).repeat(1,1,max_len)
        attention_score = torch.div(exp_attention_score, sum_attention_score + 1e-10)
        return attention_score

    def entity_aware(self, sequence, e1_mask, e2_mask):
        batch_size, max_len, feat_dim = sequence.shape
    
        e1_h = self.extract_entity(sequence, e1_mask) 
        e2_h = self.extract_entity(sequence, e2_mask)
    
        entities_h = torch.cat([e1_h,e2_h], dim=-1)
        entities_h = self.We_linear(entities_h)
        entities_expand = entities_h.unsqueeze(dim=1).repeat(1,max_len,1)
    
        s_average = torch.mean(sequence,dim = -2,keepdim = True).repeat(1,max_len,1)
        s_average = torch.cat([s_average,entities_expand],dim=-1)
    
        candidate_output = torch.cat([sequence,entities_expand],dim=-1)
        
        gate_val = torch.mul(candidate_output,s_average)
        gate_val = gate_val.view(batch_size*max_len,-1)
        gate_val = self.Wg_linear(gate_val)
        gate_val = gate_val.view(batch_size,max_len,-1)
        
        gate_val = torch.sigmoid(gate_val)

        aware_output = torch.mul(candidate_output,gate_val)      

        return aware_output

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, e1_mask=None, e2_mask=None,
                dep_adj_matrix=None, dep_type_matrix=None, valid_ids=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        if valid_ids is not None:
            valid_sequence_output = self.valid_filter(sequence_output, valid_ids)
        else:
            valid_sequence_output = sequence_output
        sequence_output = self.emb_dropout(valid_sequence_output)
        
        #add entity-aware module,new shape is (batch_size,max_length,config.hidden_size*2)
        if self.entity_hidden_size > 0:
            sequence_output = self.entity_aware(sequence_output,e1_mask,e2_mask)
        
        dep_type_embedding_outputs = self.dep_type_embedding(dep_type_matrix)
        #dep_adj_matrix = torch.clamp(dep_adj_matrix, 0, 1)
        for i, gcn_layer_module in enumerate(self.gcn_layer):
        
            attention_score = self.get_attention(sequence_output, dep_type_embedding_outputs, dep_adj_matrix, i)
            sequence_output = gcn_layer_module(sequence_output, attention_score)

            if i < len(self.gcn_layer)-1:
                sequence_output = self.gcn_dropout(sequence_output)

        e1_h = self.extract_entity(sequence_output, e1_mask)
        e2_h = self.extract_entity(sequence_output, e2_mask)
        
        # pooled_output,_ = torch.max(sequence_output,-2)
        pooled_output = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        #pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            return loss
        else:
            return logits
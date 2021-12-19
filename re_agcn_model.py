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
        self.gcn_layer = nn.ModuleList([copy.deepcopy(gcn_layer) for _ in range(config.num_gcn_layers)])
        self.ensemble_linear = nn.Linear(1, config.num_gcn_layers)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*3, config.num_labels)
        self.apply(self.init_bert_weights)
        
        #zhao_add
        linear_op = nn.Linear(config.hidden_size,config.hidden_size)
        self.linear_positive_op = nn.ModuleList([copy.deepcopy(linear_op) for _ in range(config.num_gcn_layers)])
        self.linear_reverse_op = nn.ModuleList([copy.deepcopy(linear_op) for _ in range(config.num_gcn_layers)])

        #用于将三个拼接的张量，线性转换为一个实�?        
        linear_op2 = nn.Linear(config.hidden_size*3,1)
        self.linear_op2 =nn.ModuleList([copy.deepcopy(linear_op2) for _ in range(config.num_gcn_layers)])
        #zhao_add
        
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
        #batch_size, max_len, feat_dim = val_out.shape
        #val_us = val_out.unsqueeze(dim=2)
        #val_us = val_us.repeat(1,1,max_len,1)
        #val_cat = torch.cat((val_us, dep_embed), -1)
        #atten_expand = (val_cat.float() * val_cat.float().transpose(1,2))
        #attention_score = torch.sum(atten_expand, dim=-1)
        #attention_score = attention_score / feat_dim ** 0.5
        # softmax
        #exp_attention_score = torch.exp(attention_score)
        #exp_attention_score = torch.mul(exp_attention_score.float(), adj.float())
        #sum_attention_score = torch.sum(exp_attention_score, dim=-1).unsqueeze(dim=-1).repeat(1,1,max_len)
        #attention_score = torch.div(exp_attention_score, sum_attention_score + 1e-10)
        #return attention_score
        
        #zhao_modify
        # batch_size, max_len, feat_dim = val_out.shape

        # val_out = torch.reshape(val_out,(batch_size*max_len,-1))
        # val_out = self.linear_op[i](val_out)
        
        # val_out = torch.reshape(val_out,(batch_size, max_len, -1))
        
        # val_us = val_out.unsqueeze(dim=2)
        # val_us = val_us.repeat(1,1,max_len,1)
        
        # val_cat = torch.cat((val_us,val_us.transpose(1,2),dep_embed),axis=-1)
        
        # val_cat = torch.reshape(val_cat,(batch_size*max_len*max_len,-1))
        
        # val_cat = self.linear_op2[i](val_cat)
        
        # val_cat = torch.reshape(val_cat,(batch_size, max_len, max_len,-1))
        # attention_score = val_cat.squeeze(dim=-1)
        # attention_score = F.relu(attention_score)
        
        # #softmax
        # exp_attention_score = torch.exp(attention_score)
        # exp_attention_score = torch.mul(exp_attention_score.float(), adj.float())
        # sum_attention_score = torch.sum(exp_attention_score, dim=-1).unsqueeze(dim=-1).repeat(1,1,max_len)
        # attention_score = torch.div(exp_attention_score, sum_attention_score + 1e-10)
        # return attention_score
        #在_init_(self,config)中增加线性转换结�?
        batch_size, max_len, feat_dim = val_out.shape
        
        val_us = val_out.unsqueeze(dim=2)
        val_us = val_us.repeat(1,1,max_len,1)
        
        #将hi,hj拼接
        val_cat = torch.cat((val_us,val_us.transpose(1,2)),axis=-1)
        
        #分别使用前后向转换矩阵进行线性转�?并恢复到原始维数
        val_positive = torch.reshape(val_cat,(batch_size*max_len*max_len*2,feat_dim))
        val_positive = self.linear_positive_op[i](val_positive)
        val_positive = torch.reshape(val_positive,(batch_size,max_len,max_len,2*feat_dim))
        
        val_reverse = torch.reshape(val_cat,(batch_size*max_len*max_len*2,feat_dim))
        val_reverse = self.linear_reverse_op[i](val_reverse)
        val_reverse = torch.reshape(val_reverse,(batch_size,max_len,max_len,2*feat_dim))
        
        #使用带方向的邻接矩阵对上述两个中间张量进行结�?        
        adj_reverse = torch.clamp(adj,-1,0)
        adj_positive = torch.add(adj_reverse,1)
        adj_reverse = torch.abs(adj_reverse)
        
        #扩展方向张量，以便和中间张量进行按元素乘
        adj_reverse = adj_reverse.unsqueeze(dim=-1).repeat(1,1,1,2*feat_dim)
        adj_positive = adj_positive.unsqueeze(dim=-1).repeat(1,1,1,2*feat_dim)
        
        #结合两个中间张量，获得感知方向的张量
        val_positive = (val_positive.float() * adj_positive.float())
        val_reverse = (val_reverse.float() * adj_reverse.float())
        
        val_temp = val_positive + val_reverse
        
        #将结果与依赖嵌入拼接,得到用于计算注意力的张量
        val_att = torch.cat((val_temp,dep_embed),dim=-1)
        
        #�?维张量，改变形状为二维，方便进入全连接层
        val_att = torch.reshape(val_att,(batch_size*max_len*max_len,-1))
        
        #输入到线性转换层，计算任意两个结点间的相关性置信�?        
        val_att = self.linear_op2[i](val_att)
        
        #回复到原始的4�?并删除最后一维得到注意力分�?        
        val_att = torch.reshape(val_att,(batch_size, max_len, max_len, -1))
        attention_score = val_att.squeeze(dim=-1)
        attention_score = F.relu(attention_score)
        
        #softmax
        exp_attention_score = torch.exp(attention_score)
        masked_adj = torch.abs(adj)
        exp_attention_score = torch.mul(exp_attention_score.float(), masked_adj.float())
        sum_attention_score = torch.sum(exp_attention_score, dim=-1).unsqueeze(dim=-1).repeat(1,1,max_len)
        attention_score = torch.div(exp_attention_score, sum_attention_score + 1e-10)
        return attention_score

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, e1_mask=None, e2_mask=None,
                dep_adj_matrix=None, dep_type_matrix=None, valid_ids=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        if valid_ids is not None:
            valid_sequence_output = self.valid_filter(sequence_output, valid_ids)
        else:
            valid_sequence_output = sequence_output
        sequence_output = self.dropout(valid_sequence_output)

        dep_type_embedding_outputs = self.dep_type_embedding(dep_type_matrix)
        #dep_adj_matrix = torch.clamp(dep_adj_matrix, 0, 1)
        for i, gcn_layer_module in enumerate(self.gcn_layer):
        #zhao_modify
            attention_score = self.get_attention(sequence_output, dep_type_embedding_outputs, dep_adj_matrix, i)
        #zhao_modify
            sequence_output = gcn_layer_module(sequence_output, attention_score, dep_type_embedding_outputs)
        e1_h = self.extract_entity(sequence_output, e1_mask)
        e2_h = self.extract_entity(sequence_output, e2_mask)
        
        pooled_output,_ = torch.max(sequence_output,-2)
        pooled_output = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            return loss
        else:
            return logits
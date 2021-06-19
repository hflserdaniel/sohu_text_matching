from transformers import BertModel, BertConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
from data import SentencePairDatasetWithType

# import files for NEZHA models
from NEZHA.model_nezha import NezhaConfig, NEZHAModel
from NEZHA import nezha_utils

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# basic BERT-like models
class BertClassifierSingleModel(nn.Module):
    def __init__(self, bert_dir, from_pretrained=True, task_num=2, hidden_size = 768, mid_size=512, freeze = False):
        super(BertClassifierSingleModel, self).__init__()
        self.hidden_size = hidden_size
        # could extended to multiple tasks setting, e.g. 6 classifiers for 6 subtasks
        self.task_num = task_num

        if from_pretrained:
            print("Initialize BERT from pretrained weights")
            self.bert = BertModel.from_pretrained(bert_dir)
        else:
            print("Initialize BERT from config.json, weight NOT loaded")
            self.bert_config = BertConfig.from_json_file(bert_dir+'config.json')
            self.bert = BertModel(self.bert_config)
        self.dropout = nn.Dropout(0.5)

        self.all_classifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, mid_size),
                nn.BatchNorm1d(mid_size),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(mid_size, 2)
            )
            for _ in range(self.task_num)
        ])

    def forward(self, input_ids, input_types):
        # get shared BERT model output
        mask = torch.ne(input_ids, 0)
        bert_output = self.bert(input_ids, token_type_ids=input_types, attention_mask=mask)
        cls_embed = bert_output[1]
        output = self.dropout(cls_embed)

        # get probs for two tasks A and B
        all_probs = [classifier(output) for classifier in self.all_classifier]
        return all_probs

class NezhaClassifierSingleModel(nn.Module):
    def __init__(self, bert_dir, from_pretrained=True, task_num=2, hidden_size = 768, mid_size=512, freeze = False):
        super(NezhaClassifierSingleModel, self).__init__()
        self.hidden_size = hidden_size
        self.task_num = task_num

        self.bert_config = NezhaConfig.from_json_file(bert_dir+'config.json')
        self.bert = NEZHAModel(config=self.bert_config)
        if from_pretrained:
            print("Initialize NEZHA from config.json, weight NOT loaded")
            nezha_utils.torch_init_model(self.bert, bert_dir+'pytorch_model.bin')

        self.dropout = nn.Dropout(0.5)
        self.all_classifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, mid_size),
                nn.BatchNorm1d(mid_size),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(mid_size, 2)
            )
            for _ in range(self.task_num)
        ])

    def forward(self, input_ids, input_types):
        # get shared BERT model output
        mask = torch.ne(input_ids, 0)
        bert_output = self.bert(input_ids, token_type_ids=input_types, attention_mask=mask)
        cls_embed = bert_output[1]
        output = self.dropout(cls_embed)

        # get probs for two tasks A and B
        all_probs = [classifier(output) for classifier in self.all_classifier]
        return all_probs

class SBERTSingleModel(nn.Module):
    def __init__(self, bert_dir, from_pretrained=True, task_num=2, hidden_size=768, mid_size=512, freeze = False):
        super(SBERTSingleModel, self).__init__()
        self.hidden_size = hidden_size
        self.task_num = task_num
        
        if from_pretrained:
            print("Initialize BERT from pretrained weights")
            self.bert = BertModel.from_pretrained(bert_dir)
        else:
            print("Initialize BERT from config.json, weight NOT loaded")
            self.bert_config = BertConfig.from_json_file(bert_dir+'config.json')
            self.bert = BertModel(self.bert_config)
        
        self.dropout = nn.Dropout(0.5)
        self.all_classifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size*3, mid_size),
                nn.BatchNorm1d(mid_size),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(mid_size, 2)
            )
            for _ in range(self.task_num)
        ])

    def forward(self, source_input_ids, target_input_ids):
        # 0 for [PAD], mask out the padded values
        source_attention_mask = torch.ne(source_input_ids, 0)
        target_attention_mask = torch.ne(target_input_ids, 0)
        
        # get bert output
        source_embedding = self.bert(source_input_ids, attention_mask=source_attention_mask)
        target_embedding = self.bert(target_input_ids, attention_mask=target_attention_mask)
        
        # simply take out the [CLS] represention
        # TODO: try different pooling strategies
        source_embedding = source_embedding[1]
        target_embedding = target_embedding[1]

        # concat the source embedding, target embedding and abs embedding as in the original SBERT paper
        abs_embedding = torch.abs(source_embedding-target_embedding)
        context_embedding = torch.cat([source_embedding, target_embedding, abs_embedding], -1)
        context_embedding = self.dropout(context_embedding)

        # get probs for two tasks A and B
        all_probs = [classifier(context_embedding) for classifier in self.all_classifier]
        return all_probs

class SNEZHASingleModel(nn.Module):
    def __init__(self, bert_dir, from_pretrained=True, task_num=2, hidden_size=768, mid_size=512, freeze = False):
        super(SNEZHASingleModel, self).__init__()
        self.hidden_size = hidden_size
        self.task_num = task_num
        
        self.bert_config = NezhaConfig.from_json_file(bert_dir+'config.json')
        self.bert = NEZHAModel(config=self.bert_config)
        if from_pretrained:
            print("Initialize NEZHA from config.json, weight NOT loaded")
            nezha_utils.torch_init_model(self.bert, bert_dir+'pytorch_model.bin')
        
        self.dropout = nn.Dropout(0.5)
        self.all_classifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size*3, mid_size),
                nn.BatchNorm1d(mid_size),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(mid_size, 2)
            )
            for _ in range(self.task_num)
        ])

    def forward(self, source_input_ids, target_input_ids):
        # 0 for [PAD], mask out the padded values
        source_attention_mask = torch.ne(source_input_ids, 0)
        target_attention_mask = torch.ne(target_input_ids, 0)
        
        # get bert output
        source_embedding = self.bert(source_input_ids, attention_mask=source_attention_mask)
        target_embedding = self.bert(target_input_ids, attention_mask=target_attention_mask)
        
        # simply take out the [CLS] represention
        # TODO: try different pooling strategies
        source_embedding = source_embedding[1]
        target_embedding = target_embedding[1]

        # concat the source embedding, target embedding and abs embedding as in the original SBERT paper
        abs_embedding = torch.abs(source_embedding-target_embedding)
        context_embedding = torch.cat([source_embedding, target_embedding, abs_embedding], -1)
        context_embedding = self.dropout(context_embedding)

        # get probs for two tasks A and B
        all_probs = [classifier(context_embedding) for classifier in self.all_classifier]
        return all_probs

class BertClassifierTextCNNSingleModel(nn.Module):
    def __init__(self, bert_dir, from_pretrained=True, task_num=2, hidden_size = 768, mid_size=512, freeze = False):
        super(BertClassifierTextCNNSingleModel, self).__init__()
        self.hidden_size = hidden_size
        self.task_num = task_num

        if from_pretrained:
            print("Initialize BERT from pretrained weights")
            self.bert = BertModel.from_pretrained(bert_dir)
        else:
            print("Initialize BERT from config.json, weight NOT loaded")
            self.bert_config = BertConfig.from_json_file(bert_dir+'config.json')
            self.bert = BertModel(self.bert_config)

        self.dropout = nn.Dropout(0.5)

        # for TextCNN
        filter_num = 128
        filter_sizes = [2,3,4]
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (size, hidden_size)) for size in filter_sizes])

        self.all_classifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(filter_sizes) * filter_num, mid_size),
                nn.BatchNorm1d(mid_size),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(mid_size, 2)
            )
            for _ in range(self.task_num)
        ])

    def forward(self, input_ids, input_types):
        # get shared BERT model output
        mask = torch.ne(input_ids, 0)
        bert_output = self.bert(input_ids, token_type_ids=input_types, attention_mask=mask)
        bert_hidden = bert_output[0]
        output = self.dropout(bert_hidden)

        tcnn_input = output.unsqueeze(1)
        tcnn_output = [nn.functional.relu(conv(tcnn_input)).squeeze(3) for conv in self.convs]
        # max pooling in TextCNN
        # TODO: support avg pooling
        tcnn_output = [nn.functional.max_pool1d(item, item.size(2)).squeeze(2) for item in tcnn_output]
        tcnn_output = torch.cat(tcnn_output, 1)
        tcnn_output = self.dropout(tcnn_output)

        # get probs for two tasks A and B
        all_probs = [classifier(tcnn_output) for classifier in self.all_classifier]
        return all_probs

class BertCoAttention(nn.Module):
    def __init__(self, config):
        super(BertCoAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, context_states, query_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None):
        mixed_query_layer = self.query(query_states)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.float()  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        attention_mask = extended_attention_mask

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(context_states)
            mixed_value_layer = self.value(context_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        outputs = context_layer
        return outputs

class SBERTCoAttentionModel(nn.Module):
    def __init__(self, bert_dir, from_pretrained=True, task_num=2, hidden_size=768, mid_size=512, freeze = False):
        super(SBERTCoAttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.task_num = task_num
        
        if from_pretrained:
            print("Initialize BERT from pretrained weights")
            self.bert = BertModel.from_pretrained(bert_dir)
        else:
            print("Initialize BERT from config.json, weight NOT loaded")
            self.bert_config = BertConfig.from_json_file(bert_dir+'config.json')
            self.bert = BertModel(self.bert_config)
        
        self.dropout = nn.Dropout(0.5)
        self.co_attention = BertCoAttention(hidden_size=hidden_size)
        self.all_classifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 3, mid_size),
                nn.BatchNorm1d(mid_size),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(mid_size, 2)
            )
            for _ in range(self.task_num)
        ])

    def forward(self, source_input_ids, target_input_ids):
        # 0 for [PAD], mask out the padded values
        source_attention_mask = torch.ne(source_input_ids, 0)
        target_attention_mask = torch.ne(target_input_ids, 0)
        
        # get bert output
        source_embedding = self.bert(source_input_ids, attention_mask=source_attention_mask)
        target_embedding = self.bert(target_input_ids, attention_mask=target_attention_mask)

        source_coattention_outputs = self.co_attention(target_embedding[0], source_embedding[0], source_attention_mask)
        target_coattention_outputs = self.co_attention(source_embedding[0], target_embedding[0], target_attention_mask)
        source_coattention_embedding = source_coattention_outputs[:, 0, :]
        target_coattention_embedding = target_coattention_outputs[:, 0, :]

        # simply take out the [CLS] represention
        # TODO: try different pooling strategies
        # source_embedding = source_embedding[1]
        # target_embedding = target_embedding[1]

        # concat the source embedding, target embedding and abs embedding as in the original SBERT paper
        # we also add a coattention embedding as the forth embedding
        abs_embedding = torch.abs(source_coattention_embedding - target_coattention_embedding)
        context_embedding = torch.cat([source_coattention_embedding, target_coattention_embedding, abs_embedding], -1)
        context_embedding = self.dropout(context_embedding)

        # get probs for two tasks A and B
        all_probs = [classifier(context_embedding) for classifier in self.all_classifier]
        return all_probs
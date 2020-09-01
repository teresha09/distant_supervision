# -*- coding: utf-8 -*-
# file: ian.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention, SelfAttention
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_Attention(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(LSTM_Attention, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float),freeze=False)
        self.lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention_context = SelfAttention(opt.hidden_dim, score_function='bi_linear')
        # self.pooling = nn.AvgPool1d(kernel_size=5)
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices= inputs[0]
        text_raw_len = torch.sum(text_raw_indices != 0, dim=-1)

        context = self.embed(text_raw_indices)
        # context = self.droput_context(context)
        # aspect = self.droput_aspect(aspect)
        context, (_, _) = self.lstm_context(context, text_raw_len)

        text_raw_len = torch.tensor(text_raw_len, dtype=torch.float).to(self.opt.device)
        context = torch.sum(context, dim=1)
        context = torch.div(context, text_raw_len.view(text_raw_len.size(0), 1))

        context_final = self.attention_context(context).squeeze(dim=1)

        out = self.dense(context_final)
        # out = F.softmax(out, dim=1)
        return out

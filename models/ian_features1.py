from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
import torch
import torch.nn as nn


class IANFeatures1(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(IANFeatures1, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=True)
        self.lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention_aspect = Attention(opt.hidden_dim, score_function='bi_linear')
        self.attention_context = Attention(opt.hidden_dim, score_function='bi_linear')
        self.dense = nn.Linear(opt.hidden_dim*2 + 500, opt.polarities_dim)
        self.dense2 = nn.Linear(opt.features_length, 500)
        # self.softmax = nn.LogSoftmax()

    def forward(self, inputs):
        text_raw_indices, aspect_indices, features = inputs[0], inputs[1], inputs[2]
        text_raw_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)

        context = self.embed(text_raw_indices)
        aspect = self.embed(aspect_indices)
        context, (_, _) = self.lstm_context(context, text_raw_len)
        aspect, (_, _) = self.lstm_aspect(aspect, aspect_len)

        aspect_len = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, aspect_len.view(aspect_len.size(0), 1))

        text_raw_len = torch.tensor(text_raw_len, dtype=torch.float).to(self.opt.device)
        context = torch.sum(context, dim=1)
        context = torch.div(context, text_raw_len.view(text_raw_len.size(0), 1))

        aspect_final = self.attention_aspect(aspect, context).squeeze(dim=1)
        context_final = self.attention_context(context, aspect).squeeze(dim=1)

        x = torch.cat((aspect_final, context_final), dim=-1)
        features = self.dense2(features.float())
        x = torch.cat((x, features.float()), dim=-1)
        out = self.dense(x)
        # out = self.softmax(out)
        return out
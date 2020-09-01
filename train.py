# -*- coding: utf-8 -*-
# file: train.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
import os
from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from data_utils import ABSADatesetReader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
import argparse

from models.ian_features import IAN_Features
from models.ian_no_pooling import IAN_No_Pooling
from models.lstm import LSTM
from models.ian import IAN
from models.memnet import MemNet
from models.ram import RAM
from models.td_lstm import TD_LSTM
from models.cabasc import Cabasc
from models.lstm_attention import LSTM_Attention
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

context_dict = {
    'psytar-aska': 391
}

def output_errors(corp, model_name, test_data, pred_labels, gold_labels):
    output_dir = "output/" + corp + "/errors"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    fp_out = open(output_dir + "/" + model_name + "_fp.txt", "a")
    fn_out = open(output_dir + "/" + model_name + "_fn.txt", "a")
    for data, p, l in zip(test_data, pred_labels, gold_labels):
        text = data['text'].replace('$t$', '$' + data['aspect'] + '$')
        if p == 1 and l == 0:
            fp_out.write(text + "\n")
        elif p == 0 and l == 1:
            fn_out.write(text + "\n")

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        print('> training arguments:')
        for arg in vars(opt):
            print('>>> {0}: {1}'.format(arg, getattr(opt, arg)))

        absa_dataset = ABSADatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim, max_seq_len=opt.max_seq_len, fold_num=opt.fold_num)
        self.train_data_loader = DataLoader(dataset=absa_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader = DataLoader(dataset=absa_dataset.test_data, batch_size=len(absa_dataset.test_data), shuffle=False)
        self.test_data = absa_dataset.test_data

        self.model = opt.model_class(absa_dataset.embedding_matrix, opt).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
                if len(p.shape) > 1:
                    self.opt.initializer(p)
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))

    def run(self):
        # Loss and Optimizer
        # criterion = nn.CrossEntropyLoss(torch.Tensor([ 0.70,  0.30]))
        criterion = nn.CrossEntropyLoss()
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(params, lr=self.opt.learning_rate)

        max_test_acc = 0
        global_step = 0
        all_pred = []
        all_gold = []
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(device)
                outputs = self.model(inputs)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total

                    # switch model to evaluation mode
                    self.model.eval()
                    n_test_correct, n_test_total = 0, 0
                    with torch.no_grad():
                        for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                            t_inputs = [t_sample_batched[col].to(device) for col in self.opt.inputs_cols]
                            t_targets = t_sample_batched['polarity'].to(device)
                            t_outputs = self.model(t_inputs)

                            n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                            n_test_total += len(t_outputs)
                        test_acc = n_test_correct / n_test_total
                        if test_acc > max_test_acc:
                            max_test_acc = test_acc

                    print('loss: {:.4f}, acc: {:.4f}, test acc: {:.4f}'.format(loss.item(), train_acc, test_acc))

        self.model.eval()
        gold, pred = [], []
        y_pred_pos = []
        y_pred_neg = []
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [t_sample_batched[col].to(device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(device)
                t_outputs = self.model(t_inputs)
                t_outputs = F.softmax(t_outputs)

                # n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                # n_test_total += len(t_outputs)
                # print(type(t_targets[0].item()))
                # print(t_targets[0].item())

                for t in t_targets:
                    gold.append(t.item())
                for t in torch.argmax(t_outputs, -1):
                    pred.append(t.item())
                print("Gold len = " + str(len(gold)))
                print("Pred len = " + str(len(pred)))

                # out = open("output/" + opt.dataset + "/" + opt.dataset + "_model_" + opt.model_name + "_epochs_" + str(opt.num_epoch) + "_batch_" + str(opt.batch_size) + "_lstm_" + str(opt.hidden_dim) + "_prob.txt", "a")
                for t in t_outputs:
                    y_pred_pos.append(t[1].item())
                    y_pred_neg.append(t[0].item())

            pred_str = ' '.join(str(x) for x in pred)
            gold_str = ' '.join(str(x) for x in gold)
            #
            out = open("output/result_pred.txt", "a")
            out.write(pred_str + "\n")
            out.close()


            out = open("output/result.txt", "a")
            out.write(self.opt.dataset + "\t" + str(self.opt.fold_num) + "\n")
            out.write(classification_report(gold, pred, digits=3))
            out.close()

            out = open("output/result_num.txt", "a")
            out.write(self.opt.dataset + "\t" + str(self.opt.fold_num) + "\n")
            out.write(str(metrics.precision_score(gold, pred, average='macro', labels=[0])) + "\t" + str(metrics.recall_score(gold, pred, average='macro', labels=[0])) \
                      + "\t" + str(metrics.f1_score(gold, pred, average='macro', labels=[0])) + "\t")
            out.write(str(metrics.precision_score(gold, pred, average='macro', labels=[1])) + "\t" + str(metrics.recall_score(gold, pred, average='macro', labels=[1])) \
                      + "\t" + str(metrics.f1_score(gold, pred, average='macro', labels=[1])) + "\t")
            out.write(str(metrics.precision_score(gold, pred, average='macro')) + "\t" + str(metrics.recall_score(gold, pred, average='macro')) \
                      + "\t" + str(metrics.f1_score(gold, pred, average='macro')) + "\n")
            out.close()

            print(classification_report(gold, pred, digits=3))
            print(metrics.precision_score(gold, pred, average='macro'))
            print(metrics.recall_score(gold, pred, average='macro'))
            print(metrics.f1_score(gold, pred, average='macro'))

            print(metrics.precision_score(gold, pred, average='macro', labels=[1]))
            print(metrics.recall_score(gold, pred, average='macro', labels=[1]))
            print(metrics.f1_score(gold, pred, average='macro', labels=[1]))

            print(metrics.precision_score(gold, pred, average='macro', labels=[0]))
            print(metrics.recall_score(gold, pred, average='macro', labels=[0]))
            print(metrics.f1_score(gold, pred, average='macro', labels=[0]))


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='ian', type=str)
    parser.add_argument('--dataset', default='psytar-aska', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--logdir', default='log', type=str)
    parser.add_argument('--embed_dim', default=200, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--polarities_dim', default=2, type=int)
    parser.add_argument('--hops', default=3, type=int)
    opt = parser.parse_args()

    model_classes = {
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'lstm_attention': LSTM_Attention,
        'ian_no_pooling': IAN_No_Pooling,
        'ian_features': IAN_Features,
    }
    input_colses = {
        'lstm': ['text_raw_indices'],
        'td_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'ian': ['text_raw_indices', 'aspect_indices'],
        'memnet': ['text_raw_without_aspect_indices', 'aspect_indices', 'text_left_with_aspect_indices'],
        'ram': ['text_raw_indices', 'aspect_indices'],
        'cabasc': ['text_raw_indices', 'aspect_indices', 'text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'lstm_attention': ['text_raw_indices'],
    }

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.max_seq_len = context_dict[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.model_class = model_classes[opt.model_name]
    opt.device = device

    for i in range(1,2):
        opt.fold_num = i
        ins = Instructor(opt)
        ins.run()

import dgl
import dgl.function as fn
import networkx as nx
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        
    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            #g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)


def load_data(dataset_name: str):
    if dataset_name == "cora":
        data = citegrh.load_cora()
    if dataset_name == "citeseer":
        data = citegrh.load_citeseer()
    if dataset_name == "pubmed":
        data = citegrh.load_pubmed()

    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.BoolTensor(data.train_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    g = DGLGraph(data.graph)
    return g, features, labels, train_mask, test_mask

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def bald(probs):
    ens = np.array(probs.cpu())
    predictive_entropy = entropy(np.mean(ens, axis=1))
    expected_entropy = np.mean(entropy(ens), axis=1)
    return predictive_entropy - expected_entropy

def entropy(x):
    return np.sum(-x*np.log(np.clip(x, 1e-8, 1)), axis=-1)

def max_prob(probs):
    ens = np.array(probs.cpu())
    prefiction_av = np.mean(ens, axis=1)
    max_prob = 1 - np.max(prefiction_av, 1)
    return max_prob

def accuracy(probs, labels):
    ens = np.array(probs.cpu())
    prefiction_av = np.mean(ens, axis=1)
    _, indices_ = torch.max(torch.Tensor(prefiction_av), dim=1)
    correct = torch.sum(indices_ == labels)
    acc = correct.item() * 1.0 / len(labels)
    return acc

def accuracy_with_rejection(probs, labels, ac_function = "bald", reject_percentage=10):
    reject_per_iteration = len(probs) * reject_percentage / 100 
    if ac_function == "bald":
        sorted_uncertainty = np.argsort(bald(probs))
    if ac_function == "max_prob":
        sorted_uncertainty = np.argsort(max_prob(probs))
    acc_list = []
    ens = np.array(probs.cpu())
    prefiction_av = np.mean(ens, axis=1)
    _, indices_ = torch.max(torch.Tensor(prefiction_av), dim=1)
    i_range = int(len(sorted_uncertainty) / reject_per_iteration)
    for i in range(i_range):
        new_ind = np.argsort(sorted_uncertainty)[: len(sorted_uncertainty) - i * reject_per_iteration]
        pred = indices_[new_ind]
        true_val = labels[new_ind]
        correct = torch.sum(pred == true_val)
        acc = correct.item() * 1.0 / len(true_val)
        acc_list.append(acc)
    area = np.trapz(acc_list, dx=1.0 / reject_per_iteration)
    return acc_list, area

import dgl
import dgl.function as fn
import networkx as nx
import numpy as np
import random
import time
import torch as th
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
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)


def load_data(dataset_name: str):
    if dataset_name == "cora":
        data = citegrh.load_cora()
    if dataset_name == "citeseer":
        data = citegrh.load_citeseer()
    if dataset_name == "pubmed":
        data = citegrh.load_pubmed()

    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    train_mask = th.BoolTensor(data.train_mask)
    test_mask = th.BoolTensor(data.test_mask)
    g = DGLGraph(data.graph)
    return g, features, labels, train_mask, test_mask

def set_seed(seed):
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
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

def accuracy_with_rejection(sorted_uncertainty, labels, probs, reject_per_iteration=100):
    acc_list = []
    ens = np.array(probs.cpu())
    prefiction_av = np.mean(ens, axis=1)
    _, indices_ = th.max(th.Tensor(prefiction_av), dim=1)
    i_range = int(len(sorted_uncertainty) / reject_per_iteration)
    for i in range(i_range):
        new_ind = np.argsort(sorted_uncertainty)[: len(sorted_uncertainty) - i * reject_per_iteration]
        pred = indices_[new_ind]
        true_val = labels[new_ind]
        correct = th.sum(pred == true_val)
        acc = correct.item() * 1.0 / len(true_val)
        acc_list.append(acc)
    return acc_list
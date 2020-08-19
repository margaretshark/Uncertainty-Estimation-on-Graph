import numpy as np
import random
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh


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


class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GATLayer, self).__init__()
        # equation (1)
        self.linear = nn.Linear(in_feats, out_feats, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_feats, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = th.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = th.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, feature):
        # equation (1)
        z = self.linear(feature)
        g.ndata['z'] = z
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim))
        self.merge = merge

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return th.cat(head_outs, dim=1)
        else:
            # merge using average
            return th.mean(th.stack(head_outs))


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
    _, indices_ = th.max(th.Tensor(prefiction_av), dim=1)
    correct = th.sum(indices_ == labels)
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


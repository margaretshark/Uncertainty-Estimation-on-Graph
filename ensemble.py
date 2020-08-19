import torch as th
import torch.nn as nn
import torch.nn.functional as F
import graph
import utils
import numpy as np


class Ensemble(nn.Module):

    def __init__(self, num_of_nets: int=10, graph_type: str="GCN", dataset_name: str="cora", seed: int=42):
        super(Ensemble, self).__init__()
        self.models = []
        for i in range(num_of_nets):
            net = graph.Net(graph_type=graph_type, dataset_name=dataset_name, seed=seed + i)
            net.graph.add_edges(net.graph.nodes(), net.graph.nodes())
            net.train_loop()
            self.models.append(net)

    def forward(self):
        outs = []
        for model in self.models:
            logits, labels = model.evaluate()
            outs.append(logits.unsqueeze(1))
        out = th.cat(outs, dim=1)
        probs = F.softmax(out, dim=-1)
        return probs, labels
    

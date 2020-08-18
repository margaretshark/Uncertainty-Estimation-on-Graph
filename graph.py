import numpy as np
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import utils


class Net(nn.Module):

    def __init__(self, graph_type: str="GCN", dataset_name: str="cora", seed: int=42):
        utils.set_seed(seed)
        if dataset_name == "cora":
            self.num_classes = 7
        
        if dataset_name == "citeseer":
            self.num_classes = 6
        
        if dataset_name == "pubmed":
            self.num_classes = 3
        
        self.graph, self.features, self.labels, self.train_mask, self.test_mask = utils.load_data(dataset_name)
        self.features_dimension = len(self.features[0])

        if graph_type == "GCN":
            super(Net, self).__init__()
            self.layer1 = utils.GCNLayer(self.features_dimension, 16)
            self.layer2 = utils.GCNLayer(16, self.num_classes)

    def forward(self):
        x = F.relu(self.layer1(self.graph, self.features))
        x = self.layer2(self.graph, x)
        return x

    def train_loop(self, num_epoch: int=50, lr: float=1e-2):
        dur = []
        optimizer = th.optim.Adam(self.parameters(), lr=1e-2)
        for epoch in range(num_epoch):
            if epoch >=3:
                t0 = time.time()
            self.train()
            logits = self()
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp[self.train_mask], self.labels[self.train_mask])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch >=3:
                dur.append(time.time() - t0)
                
            acc = calculate_accuracy(*self.evaluate())
            
            print("Epoch {:05d} | loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), acc, np.mean(dur)))

    def evaluate(self):
        self.eval()
        with th.no_grad():
            logits = self()
            logits = logits[self.test_mask]
            labels = self.labels[self.test_mask]
            return logits, labels


def calculate_accuracy(logits, labels):
    _, indices = th.max(logits, dim=1)
    correct = th.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

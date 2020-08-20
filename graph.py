import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from pytorchtools import EarlyStopping


class Net(nn.Module):

    def __init__(self, params, seed: int=42):
        super(Net, self).__init__()
        utils.set_seed(seed)
        if params["dataset_name"] == "cora":
            self.num_classes = 7
        
        if params["dataset_name"] == "citeseer":
            self.num_classes = 6
        
        if params["dataset_name"] == "pubmed":
            self.num_classes = 3
        
        self.graph, self.features, self.labels, self.train_mask, self.test_mask = utils.load_data(params["dataset_name"])
        self.features_dimension = len(self.features[0])

        if params["graph_type"] == "GCN":
            self.layer1 = utils.GCNLayer(self.features_dimension, 16)
            self.layer2 = utils.GCNLayer(16, self.num_classes)
        
        if params["graph_type"] == "GAT":
            self.layer1 = utils.MultiHeadGATLayer(self.features_dimension, 8, 2)
            self.layer2 = utils.MultiHeadGATLayer(16, self.num_classes, 1)

    def forward(self):
        x = F.relu(self.layer1(self.graph, self.features))
        x = self.layer2(self.graph, x)
        return x

    def train_model(self, optimizer, patience, n_epochs):
        lr_scheduler = optimizer.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1)
        dur = []
        train_losses = []
        valid_losses = []
        avg_train_losses = []
        avg_valid_losses = []
        epoch_train_loss = []

        early_stopping = EarlyStopping(patience=patience, verbose=True)
        for epoch in range(n_epochs):
            if epoch >= 3:
                t0 = time.time()

            self.train()
            logits = self()
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp[self.train_mask], self.labels[self.train_mask])
            epoch_train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            acc = calculate_accuracy(*self.evaluate())
            epoch_train_loss_mean = np.mean(epoch_train_loss)
            lr_scheduler.step(epoch_train_loss_mean)
            print("Epoch {:05d} | loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), acc, np.mean(dur)))

            with torch.no_grad():
                logits = self(self.graph, self.features)
                logp = F.log_softmax(logits, 1)
                loss_val = F.nll_loss(logp[self.test_mask], self.labels[self.test_mask])
                valid_losses.append(loss_val.item())

            valid_loss = np.average(valid_losses)
            epoch_len = len(str(n_epochs))

            train_losses = []
            valid_losses = []

            early_stopping(valid_loss, self)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        return self, avg_train_losses, avg_valid_losses

    def evaluate(self):
        self.eval()
        with torch.no_grad():
            logits = self()
            logits = logits[self.test_mask]
            labels = self.labels[self.test_mask]
            return logits, labels


def calculate_accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

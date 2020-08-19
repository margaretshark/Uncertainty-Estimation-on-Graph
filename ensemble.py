import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Ensemble(nn.Module):

    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models

    def forward(self):
        outs = []
        for model in self.models:
            logits, labels = model.evaluate()
            outs.append(logits.unsqueeze(1))
        out = th.cat(outs, dim=1)
        probs = F.softmax(out, dim=-1)
        return probs, labels
    

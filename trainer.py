import graph
import os
import torch as th
import torch.nn as nn

class Trainer(nn.Module):
    def __init__(self, params, model_dir: str = "models", seed: int = 42):
        super(Trainer, self).__init__()
        self.model_dir = model_dir
        self.seed = seed
        self.params = params
        self.models = []

    def forward(self):
        if os.path.exists(self.model_dir):
            for i in range(self.params["num_of_nets"]):
                net = graph.Net(self.params, self.seed + i)
                filename = f"{self.model_dir}/checkpoint_{i}.pth"
                net.load_state_dict(th.load(filename))
                self.models.append(net)
        else:
            os.mkdir(self.model_dir)
            for i in range(self.params["num_of_nets"]):
                net = graph.Net(self.params, self.seed + i)
                net.graph.add_edges(net.graph.nodes(), net.graph.nodes())
                net.train_model()
                self.models.append(net)
                th.save(net.state_dict(), f"{self.model_dir}/checkpoint_{i}.pth")
        return self.models

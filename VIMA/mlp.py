import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class Heads(nn.Module):
    def __init__(self, in_features, num_heads):
        super().__init__()
        heads = {}

        # list of linear heads with output dimentions [1, num_heads]
        for i in range (num_heads):
            with torch.no_grad():
                head = nn.Linear(in_features=in_features, out_features=i+1, bias=False)
                heads[str(i)]=head
        self.heads = nn.ParameterDict(heads)

    # key of heads must be string
    def forward(self, x, idx):
        t = self.heads[str(idx)]
        x = t(x)
        return x

class MLPWithHeads(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__() 
        self.heads = Heads(in_features=hidden_size, num_heads=output_size)

        self.fc1 = nn.Linear(input_size, hidden_size//2, bias=False)
        self.fc2 = nn.Linear(hidden_size//2, hidden_size//2, bias=False)
        self.fc3 = nn.Linear(hidden_size//2, hidden_size, bias=False)

    def forward(self, x, num_embeddings):
        x = x.flatten()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        head_idx = num_embeddings-1  
        with torch.no_grad():
            x =self.heads(x, head_idx)
        return x

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.flatten()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.reshape(30, 1, 768)
        return x


if __name__ == '__main__':
    t = torch.ones((30, 1, 768))
    t = t.flatten()

    mlp = MLPWithHeads(input_size=23040, hidden_size=269, output_size=30)
    model_info = summary(mlp)

    out = mlp(t, num_embeddings=9)
    print(out.shape)   
    out.shape

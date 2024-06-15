import torch
from mlp import MLP

class ModelBuilder():
    def __init__(self, model_type=None):
        self.model = None
        self.name = model_type

        if model_type == "mlp":
            self.model = MLP(768, 1024, 768)
            checkpoint = torch.load("MLP_2.pth", map_location=torch.device('cuda'))
            self.model.fc1.weight.data = checkpoint['fc1.weight']
            self.model.fc1.bias.data = checkpoint['fc1.bias']
            self.model.fc2.weight.data = checkpoint['fc2.weight']
            self.model.fc2.bias.data = checkpoint['fc2.bias']
            self.model.fc3.weight.data = checkpoint['fc3.weight']
            self.model.fc3.bias.data = checkpoint['fc3.bias']

    def return_prediciton(self, prompt_tokens):
        self.model.eval()
        
        if self.name == "mlp":
            prompt_tokens = prompt_tokens.view(-1, 768)
            output_tokens = self.model(prompt_tokens)
            output_tokens = output_tokens.unsqueeze(1)
            return output_tokens
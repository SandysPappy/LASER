import torch
from mlp.mlp_pad import MLP as MLP_Pad
from mlp.mlp import MLP as MLP_UnPad

class ModelBuilder():
    def __init__(self, model_type=None):
        self.model = None
        self.name = model_type

        if model_type == "mlp_unpad":
            self.model = MLP_UnPad(768, 128, 768)
            checkpoint = torch.load("MLP_5.pth", map_location=torch.device('cpu'))
            self.model.fc1.weight.data = checkpoint['fc1.weight']
            self.model.fc1.bias.data = checkpoint['fc1.bias']
            self.model.fc2.weight.data = checkpoint['fc2.weight']
            self.model.fc2.bias.data = checkpoint['fc2.bias']
            self.model.fc3.weight.data = checkpoint['fc3.weight']
            self.model.fc3.bias.data = checkpoint['fc3.bias']

        if model_type == "mlp_pad":
            self.model = MLP_Pad(input_size=30*1*768, hidden_size=269, output_size=30*1*768)
            checkpoint = torch.load("MLP_5.pth", map_location=torch.device('cpu'))
            self.model.fc1.weight.data = checkpoint['fc1.weight']
            self.model.fc1.bias.data = checkpoint['fc1.bias']
            self.model.fc2.weight.data = checkpoint['fc2.weight']
            self.model.fc2.bias.data = checkpoint['fc2.bias']
            self.model.fc3.weight.data = checkpoint['fc3.weight']
            self.model.fc3.bias.data = checkpoint['fc3.bias']

    def return_prediciton(self, prompt_tokens):
        self.model.eval()
        
        if self.name == "mlp_unpad":
           tokens = []
           for i in range(prompt_tokens.size(0)):
               prompt_token = prompt_tokens[i, :, :]
               output_token = self.model(prompt_token)
               tokens.append(output_token)

           combined_tokens = torch.stack(tokens, dim=0)
           return combined_tokens
         
        if self.name == "mlp_pad":
            test_embed = prompt_tokens
            test_embed = test_embed.flatten()
            test_embed = test_embed.unsqueeze(dim=0)

            output = self.model(test_embed) #(b, 30*1*768)
            output = output.squeeze(dim=0)
            output = output.reshape(30,1,768)              
            return output
             

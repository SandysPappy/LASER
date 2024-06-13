import torch
from mlp.mlp_pad import MLP_Pad
from mlp.mlp import MLP as MLP_UnPad

class ModelBuilder():
    def __init__(self, model_type=None):
        self.model = None
        self.name = model_type

        if model_type == "mlp_unpad":
            self.model = MLP_UnPad(768, 1024, 768)
            checkpoint = torch.load("MLP_2.pth", map_location=torch.device('cuda'))
            self.model.fc1.weight.data = checkpoint['fc1.weight']
            self.model.fc1.bias.data = checkpoint['fc1.bias']
            self.model.fc2.weight.data = checkpoint['fc2.weight']
            self.model.fc2.bias.data = checkpoint['fc2.bias']
            self.model.fc3.weight.data = checkpoint['fc3.weight']
            self.model.fc3.bias.data = checkpoint['fc3.bias']

        if model_type == "mlp_pad":
            self.model = MLP_Pad(input_size=30*1*768, hidden_size=269, output_size=30*1*768)
            checkpoint = torch.load("model_weight_20epoch_32batch_noShuffle_pad0_noflat.pth", map_location=torch.device('cpu'))
            self.model.fc1.weight.data = checkpoint['fc1.weight']
            self.model.fc1.bias.data = checkpoint['fc1.bias']
            self.model.fc2.weight.data = checkpoint['fc2.weight']
            self.model.fc2.bias.data = checkpoint['fc2.bias']
            self.model.fc3.weight.data = checkpoint['fc3.weight']
            self.model.fc3.bias.data = checkpoint['fc3.bias']

    def return_prediciton(self, prompt_tokens):
        self.model.eval()
        
        if self.name == "mlp_unpad":
        #    tokens = []
        #    for i in range(prompt_tokens.size(0)):
        #        prompt_token = prompt_tokens[i, :, :]
        #        output_token = self.model(prompt_token)
        #        tokens.append(output_token)

        #    combined_tokens = torch.stack(tokens, dim=0)
        #    return combined_tokens
            return self.model(prompt_tokens)
         
        if self.name == "mlp_pad":

            if torch.cuda.is_available():
               device = 'cuda'
            else:
                device = 'cpu'

            # calculate how many zeros to pad
            token_len, _, _, = prompt_tokens.shape
            padding_len = 30 - token_len

            # turn into shape (n + padding, 1, 768)
            padding = torch.zeros((padding_len, 1, 768), device=device)
            #padding.to('cuda')
            #prompt_tokens.to(device)
            #print(f'Padding is using: {padding.device}')
            #print(f'Prompt tokens is using: {padding.device}')
            test_embed = torch.cat((prompt_tokens, padding), dim=0)

            test_embed = test_embed.flatten()
            test_embed = test_embed.unsqueeze(dim=0)

            self.model.to(device)
            output = self.model(test_embed) #(b, 30*1*768)
            output = output.squeeze(dim=0)
            
            output = output.reshape(30,1,768)
            output = output[:token_len, :, :]

            return output
             

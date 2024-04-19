import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset_builder import get_laser_dataset
from mlp.mlp_pad import MLPWithHeads
from mlp.mlp_pad import MLP
from tqdm import tqdm
from torch.nn.functional import normalize
import numpy as np
import torch.nn.functional as F

# Example usage
if __name__ == "__main__":

    output_path="mlp_save_weight/save_model_weight.pt"
    lr = 1e-3
    num_epochs = 10
    batch_size = 32

    #model = MLPWithHeads(input_size=23040*batch_size, hidden_size=269, output_size=30)
    model = MLP(input_size=30*1*768, hidden_size=269, output_size=30*1*768, batch=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = get_laser_dataset(task="all", partition="all")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
   
    train_loss = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        pbar = tqdm(dataloader, total=len(dataloader))
        batch_loss=[]
        for batch in pbar:
            if batch is None:
                continue
            optimizer.zero_grad()
            # Access the first tensor in the batch
            success = batch["success"]
            success = success.bool()

            base_embedding = batch["base_embeddings"]
            #base_embedding = F.normalize(base_embedding)
            base_embedding = base_embedding.to(device)
            attack_embedding = batch["attack_embeddings"]
            #attack_embedding = F.normalize(attack_embedding)
            attack_embedding = attack_embedding.to(device)
            
            num_of_embeddings = batch["num_of_embeddings"]
            num_of_embeddings = num_of_embeddings.int()
            # forward pass
            embedding_output = model(attack_embedding)
            #embedding_output = F.normalize(embedding_output)
            loss = criterion(embedding_output, base_embedding)
            
            # backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_loss.append(loss.item())
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        batch_loss = np.array(batch_loss)
        train_loss.append(batch_loss.mean())
        # Print the values
        # print("Base Embeddings shape:", base_embedding)
        # print("Attack Embeddings shape:", attack_embedding)
        # print("Number of Embeddings:", num_of_embeddings)

        # # If we want to grab the unpadded embeddings
        # first_embeddings_base = batch["base_embeddings"][0][:num_of_embeddings]
        # first_embeddings_attack = batch["attack_embeddings"][0][:num_of_embeddings]

        # print("Original base embeddings: ", first_embeddings_base.shape)
        # print("Original base embeddings: ", first_embeddings_base)

        # print("Original attack embeddings: ",
        #       first_embeddings_attack.shape)
        # print("Original attack embeddings: ", first_embeddings_attack)

        # break  # Exit loop after printing the first tensor
    
    print("training loss:", train_loss)
    # save model
    torch.save(model.state_dict(), output_path)





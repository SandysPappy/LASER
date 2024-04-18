import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_builder import get_laser_dataset
from mlp import MLP


# Example usage
if __name__ == "__main__":

    lr = 1e-3
    num_epochs = 10
    model = MLP(input_size=768, hidden_size=256,
                output_size=768)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = get_laser_dataset(task="all", partition="all")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    num_batches = 0

    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Access the first tensor in the batch
            success = bool(batch["success"][0])
            base_embedding = batch["base_embeddings"][0]
            attack_embedding = batch["attack_embeddings"][0]
            num_of_embeddings = int(batch["num_of_embeddings"][0])

            # forward pass
            embedding_output = model(attack_embedding)
            loss = criterion(embedding_output, 256)

            # backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

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

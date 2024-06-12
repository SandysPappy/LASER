import torch
from torch.utils.data import DataLoader
from dataset_builder import get_laser_dataset
import torch.nn as nn
import torch.optim as optim
from mlp import MLP
from tqdm import tqdm

# Example usage
if __name__ == "__main__":

    hidden_size = 128
    output_size = 768  # Binary classification
    model = MLP(768, hidden_size, output_size)
    model = model.to("cuda")

    # Define loss function and optimizer
    criterion = nn.MSELoss()

    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare dataset and data loader
    dataset = get_laser_dataset(task="all", partition="all")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model.train(True)
    num_epochs = 1

    for epoch in range(num_epochs):
        total_similarity_epoch = 0
        total_samples_epoch = 0
        total_loss_epoch = 0


        pbar = tqdm(dataloader, total=len(dataloader))

        for batch in pbar:
        #[B,1,768]
            total_similarity_batch = 0
            total_samples_batch = 0
            total_loss_batch = 0
            
            optimizer.zero_grad()  # Clear gradients at the start of each batch

            for i in range(len(list(batch.values())[0])):
                num_of_embeddings = batch["num_of_embeddings"][i]
                attack_embeddings = batch["attack_embeddings"][i][:num_of_embeddings].to("cuda")
                base_embeddings = batch["base_embeddings"][i][:num_of_embeddings].to("cuda")

                # embedding shape [token, 1, 768]

                # reshape to [token, 768]

                attack_embeddings = attack_embeddings.view(-1, 768)
                base_embeddings = base_embeddings.view(-1, 768)

                embedding_output = model(attack_embeddings)

                # Compute similarity between the output embeddings and the base embeddings

                similarity = torch.cosine_similarity(embedding_output, base_embeddings, dim=1)

                # returns tensor of shape [token]

                total_similarity_batch += similarity.sum().item() # Accumulate similarity for the batch

                total_samples_batch += num_of_embeddings

                loss = criterion(embedding_output, base_embeddings)

                loss.backward()  # Compute gradients

                total_loss_batch += loss.item()  # Accumulate loss for the batch

                pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")


            optimizer.step()  # Update model parameters after processing the entire batch

            average_similarity_batch = total_similarity_batch / total_samples_batch
            average_loss_batch = total_loss_batch / len(list(batch.values())[0])

            total_similarity_epoch += total_similarity_batch
            total_samples_epoch += total_samples_batch
            total_loss_epoch += total_loss_batch

        # Compute average similarity and loss for the epoch
        average_similarity_epoch = total_similarity_epoch / total_samples_epoch
        average_loss_epoch = total_loss_epoch / total_samples_epoch
        print(f"Average similarity for epoch {epoch+1}: {average_similarity_epoch}")
        print(f"Average loss for epoch {epoch+1}: {average_loss_epoch}")

    # Save the trained model
    # torch.save(model.state_dict(), '/home/cap6412.student10/Vima/MLP_2.pth')

import torch
from torch.utils.data import DataLoader
from dataset_builder import get_laser_dataset
import torch.nn as nn
import torch.optim as optim
from mlp import MLP
from tqdm import tqdm
import matplotlib.pyplot as plt
# Example usage
if __name__ == "__main__":

    hidden_size = 1024
    output_size = 768  # Binary classification
    model = MLP(768, hidden_size, output_size)
    model = model.to("cuda")

    # Define loss function and optimizer
    criterion = nn.MSELoss()

    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Prepare dataset and data loader
    dataset = get_laser_dataset(task="all", partition="all")
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    model.train(True)
    num_epochs = 5

    losses = []
    similarities = []
    lrs = []

    for epoch in range(num_epochs):
        total_similarity_epoch = 0
        total_samples_epoch = 0
        total_loss_epoch = 0


        pbar = tqdm(dataloader, total=len(dataloader))

        for attack_embs, base_embs in pbar:

            attack_embs = attack_embs.to("cuda")
            base_embs = base_embs.to("cuda")

            optimizer.zero_grad()  # Clear gradients at the start of each batch

            embedding_output = model(attack_embs)

            # Compute similarity between the output embeddings and the base embeddings  
            similarity = torch.nn.functional.cosine_similarity(embedding_output, base_embs, dim=1)

            loss = criterion(embedding_output, base_embs)

            loss.backward()  # Compute gradients

            optimizer.step()  # Update model parameters after processing the entire batch

            losses.append(loss.item())
            similarities.append(similarity.mean().item())

            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f} Similarity: {similarity.mean().item():.4f}")

            total_similarity_epoch += similarity.sum().item()  # Accumulate similarity for the epoch
            total_samples_epoch += len(similarity)
            total_loss_epoch += loss.item()

        # scheduler.step()  # Update learning rate

        # Compute average similarity and loss for the epoch
        average_similarity_epoch = total_similarity_epoch / total_samples_epoch
        average_loss_epoch = total_loss_epoch / len(dataloader)
        print(f"Average similarity for epoch {epoch+1}: {average_similarity_epoch}")
        print(f"Average loss for epoch {epoch+1}: {average_loss_epoch}")

    ax, fig = plt.subplots(1, 2, figsize=(15, 5))
    fig[0].plot(losses)
    fig[0].set_title("Loss")
    fig[0].set_xlabel("Batch")
    fig[0].set_ylabel("Loss")

    fig[1].plot(similarities)
    fig[1].set_title("Similarity")
    fig[1].set_xlabel("Batch")
    plt.show()

    

    # Save the trained model
    torch.save(model.state_dict(), 'MLP_2.pth')

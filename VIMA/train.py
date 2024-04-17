import torch
from torch.utils.data import DataLoader
from dataset_builder import get_laser_dataset

# Example usage
if __name__ == "__main__":

    dataset = get_laser_dataset(task="all", partition="all")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in dataloader:
        # Access the first tensor in the batch
        base_embedding = batch["base_embeddings"][0].shape
        attack_embedding = batch["attack_embeddings"][0].shape
        num_of_embeddings = int(batch["num_of_embeddings"][0])
        success = bool(batch["success"][0])
        base_prompt = batch["base_prompt"][0]
        attack_prompt = batch["attack_prompt"][0]
        rephrasings = batch["rephrasings"][0]
        seed = int(batch["seed"][0])
        partition = batch["partition"][0]
        task = batch["task"][0]

        # Print the values
        print("Base Embeddings shape:", base_embedding)
        print("Attack Embeddings shape:", attack_embedding)
        print("Number of Embeddings:", num_of_embeddings)
        print("Success:", success)
        print("Base Prompt:", base_prompt)
        print("Attack Prompt:", attack_prompt)
        print("Rephrasings:", rephrasings)
        print("Seed:", seed)
        print("Partition:", partition)
        print("Task:", task)

        # If we want to grab the unpadded embeddings
        first_embeddings_base = batch["base_embeddings"][0][:num_of_embeddings]
        first_embeddings_attack = batch["attack_embeddings"][0][:num_of_embeddings]

        print("Original base embeddings: ", first_embeddings_base.shape)
        print("Original base embeddings: ", first_embeddings_base)

        print("Original attack embeddings: ", first_embeddings_attack.shape) 
        print("Original attack embeddings: ", first_embeddings_attack) 

        break # Exit loop after printing the first tensor
import torch
from torch.utils.data import Dataset

#Add your file path
attack_file_path = "LASER_Dataset/attack_prompts/"
base_file_path = "LASER_Dataset/base_prompts/"

laser_dataset = {
    # "combinatorial/base_combinatorial_generalization_rearrange_42_dataset.pt": [
    # #"combinatorial/attack_combinatorial_generalization_Color Rephrase_rearrange_42_dataset.pt",
    # "combinatorial/attack_combinatorial_generalization_Extend_rearrange_42_dataset.pt",
    # "combinatorial/attack_combinatorial_generalization_Noun_rearrange_42_dataset.pt",
    # # "combinatorial/attack_combinatorial_generalization_Stealth_rearrange_42_dataset.pt" 
    # ],
    # "combinatorial/base_combinatorial_generalization_scene_understanding_42_dataset.pt": [
    # #"combinatorial/attack_combinatorial_generalization_Color Rephrase_scene_understanding_42_dataset.pt",
    # "combinatorial/attack_combinatorial_generalization_Extend_scene_understanding_42_dataset.pt",
    # "combinatorial/attack_combinatorial_generalization_Noun_scene_understanding_42_dataset.pt",
    # # "combinatorial/attack_combinatorial_generalization_Stealth_scene_understanding_42_dataset.pt" 
    # ],
    # "combinatorial/base_combinatorial_generalization_visual_manipulation_42_dataset.pt" : [
    # #"combinatorial/attack_combinatorial_generalization_Color Rephrase_visual_manipulation_42_dataset.pt",
    # "combinatorial/attack_combinatorial_generalization_Extend_visual_manipulation_42_dataset.pt",
    # "combinatorial/attack_combinatorial_generalization_Noun_visual_manipulation_42_dataset.pt",
    # # "combinatorial/attack_combinatorial_generalization_Stealth_visual_manipulation_42_dataset.pt"
    # ],
    "placement/base_placement_generalization_rearrange_42_dataset.pt" : [
    #"placement/attack_placement_generalization_Color Rephraserearrange_42_dataset.pt",
    "placement/attack_placement_generalization_Extendrearrange_42_dataset.pt",
   # "placement/attack_placement_generalization_Nounrearrange_42_dataset.pt",
    # "placement/attack_placement_generalization_Stealth_rearrange_42_dataset.pt"
    ],
    "placement/base_placement_generalization_scene_understanding_42_dataset.pt": [
    #"placement/attack_placement_generalization_Color Rephrasescene_understanding_42_dataset.pt",
    "placement/attack_placement_generalization_Extendscene_understanding_42_dataset.pt",
   # "placement/attack_placement_generalization_Nounscene_understanding_42_dataset.pt",
    # "placement/attack_placement_generalization_Stealth_scene_understanding_42_dataset.pt"
    ],
    "placement/base_placement_generalization_visual_manipulation_42_dataset.pt": [
    #"placement/attack_placement_generalization_Color Rephrasevisual_manipulation_42_dataset.pt",
    "placement/attack_placement_generalization_Extendvisual_manipulation_42_dataset.pt",
    # "placement/attack_placement_generalization_Nounvisual_manipulation_42_dataset.pt",
    # "placement/attack_placement_generalization_Stealth_visual_manipulation_42_dataset.pt"
    ],
}

# Exmaples
# dataset = get_laser_dataset(task="visual_manipulation", partition="placement_generalization")
# dataset = get_laser_dataset(task="all", partition="all")
def get_laser_dataset(task=None, partition=None, pad_len = 30):
    if task == None or partition == None:
        return KeyError("Include correct dataset task and partition")
    
    matching_keys = [key for key in laser_dataset if task in key and partition in key]
    if matching_keys:
        key_to_use = matching_keys[0]  # Using the first matching key
        subset_dataset = {key_to_use: laser_dataset[key_to_use]}
        return PromptEmbeddingsDataset(subset_dataset, pad_len)
    
    if task == "all" and partition == "all":
        return PromptEmbeddingsDataset(laser_dataset, pad_len)
    
    return KeyError("Task or partition not found in the laser_dataset")

class PromptEmbeddingsDataset(Dataset):
    def __init__(self, dataset, pad_len):
        self.pad_len = pad_len

        self.base_embeddings = []
        self.attack_embeddings = []

        for base_dataset, attack_datasets in dataset.items():
            #Load base file
            base = torch.load(base_file_path + base_dataset)

            for attack_dataset in attack_datasets:
                if "attack" in attack_dataset:
                    attack = torch.load(attack_file_path + attack_dataset)
                elif "base" in attack_dataset:
                    attack = torch.load(base_file_path + attack_dataset)

                attack_size = len(attack)

                # Loop through base and attack file
                for i in range(attack_size):
                    for j in range(150):
                        attack_emb, base_emb = attack[i][j]["embedding"], base[0][j]["embedding"]
                        success = attack[i][j]["success"]

                        if attack_emb.shape == base_emb.shape:

                            # Attack - Base pairs
                            for attack_token, base_token in zip(attack_emb, base_emb):
                                self.base_embeddings.append(base_token[0])
                                self.attack_embeddings.append(attack_token[0])

                            # Base - Base pairs, to make sure the model doesn't drift from the base embeddings
                            for token in base_emb:
                                self.base_embeddings.append(token[0])
                                self.attack_embeddings.append(token[0])

                        else:
                            # Handle the case where the embeddings have different shapes (optional)
                            print(f"Warning: Embeddings at index ({i}, {j}) have different shapes.")

    def __len__(self):
        return len(self.base_embeddings)

    def __getitem__(self, idx):
        base_embeddings = self.base_embeddings[idx]
        attack_embeddings = self.attack_embeddings[idx]

        return attack_embeddings, base_embeddings

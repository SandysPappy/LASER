import torch
import pandas as pd
import numpy as np
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

#Add your file path
attack_file_path = "LASER_Dataset/attack_prompts/"
base_file_path = "LASER_Dataset/base_prompts/"

laser_dataset = {
    "combinational/base_combinatorial_generalization_rearrange_42_dataset.pt": [
    "combinatorial/attack_combinatorial_generalization_Color Rephrase_rearrange_42_dataset.pt",
    "combinatorial/attack_combinatorial_generalization_Extend_rearrange_42_dataset.pt",
    "combinatorial/attack_combinatorial_generalization_Noun_rearrange_42_dataset.pt"  
    ],
    "combinational/base_combinatorial_generalization_scene_understanding_42_dataset.pt": [
    "combinatorial/attack_combinatorial_generalization_Color Rephrase_scene_understanding_42_dataset.pt",
    "combinatorial/attack_combinatorial_generalization_Extend_scene_understanding_42_dataset.pt",
    "combinatorial/attack_combinatorial_generalization_Noun_scene_understanding_42_dataset.pt" 
    ],
    "combinational/base_combinatorial_generalization_visual_manipulation_42_dataset.pt" : [
    "combinatorial/attack_combinatorial_generalization_Color Rephrase_visual_manipulation_42_dataset.pt",
    "combinatorial/attack_combinatorial_generalization_Extend_visual_manipulation_42_dataset.pt",
    "combinatorial/attack_combinatorial_generalization_Noun_visual_manipulation_42_dataset.pt"
    ],
    "placement/base_placement_generalization_rearrange_42_dataset.pt" : [
    "placement/attack_placement_generalization_Color Rephraserearrange_42_dataset.pt",
    "placement/attack_placement_generalization_Extendrearrange_42_dataset.pt",
    "placement/attack_placement_generalization_Nounrearrange_42_dataset.pt"
    ],
    "placement/base_placement_generalization_scene_understanding_42_dataset.pt": [
    "placement/attack_placement_generalization_Color Rephrasescene_understanding_42_dataset.pt",
    "placement/attack_placement_generalization_Extendscene_understanding_42_dataset.pt",
    "placement/attack_placement_generalization_Nounscene_understanding_42_dataset.pt"
    ],
    "placement/base_placement_generalization_visual_manipulation_42_dataset.pt": [
    "placement/attack_placement_generalization_Color Rephrasevisual_manipulation_42_dataset.pt",
    "placement/attack_placement_generalization_Extendvisual_manipulation_42_dataset.pt",
    "placement/attack_placement_generalization_Nounvisual_manipulation_42_dataset.pt"
    ]
}

def get_laser_dataset(task="", partition="", pad_len = 30):
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
        self.success_labels = []
        self.num_of_embeddings = []
        self.base_prompt = []
        self.attack_prompt = []
        self.task = []
        self.partition = []
        self.seed = []
        self.rephrasings = []

        for base_dataset, attack_datasets in dataset.items():
            #Load base file
            base = torch.load(base_file_path + base_dataset)
            for attack_dataset in attack_datasets:
                # Load attack file
                attack = torch.load(attack_file_path + attack_dataset)

                # Loop through base and attack file
                for i in range(9):
                    for j in range(150):
                        attack_emb, base_emb = attack[i][j]["embedding"], base[0][j]["embedding"]
                        success = attack[i][j]["success"]

                        if attack_emb.shape == base_emb.shape:
                            self.base_embeddings.append(base_emb)
                            self.attack_embeddings.append(attack_emb)
                            self.success_labels.append(success)
                            self.num_of_embeddings.append(int(attack_emb.shape[0]))
                            self.base_prompt.append(attack[i][j]["base_prompt"])
                            self.attack_prompt.append(attack[i][j]["attack_prompt"])
                            self.rephrasings.append(attack[i][j]["rephrasings"])
                            self.seed.append(attack[i][j]["seed"])
                            self.partition.append(attack[i][j]["partition"])
                            self.task.append(attack[i][j]["task"])
                        else:
                            # Handle the case where the embeddings have different shapes (optional)
                            print(f"Warning: Embeddings at index ({i}, {j}) have different shapes.")

    def __len__(self):
        return len(self.base_embeddings)

    def __getitem__(self, idx):
        base_embeddings = self.base_embeddings[idx]
        attack_embeddings = self.attack_embeddings[idx]
        success = self.success_labels[idx]
        num_of_embeddings = self.num_of_embeddings[idx]
        base_prompt = self.base_prompt[idx]
        attack_prompt = self.attack_prompt[idx]
        rephrasings = self.rephrasings[idx]
        seed = self.seed[idx]
        partition = self.partition[idx]
        task = self.task[idx]

        base_embeddings = torch.nn.functional.pad(base_embeddings, (0, 0, 0, 0, 0, self.pad_len - num_of_embeddings), value=-1)
        attack_embeddings = torch.nn.functional.pad(attack_embeddings, (0, 0, 0, 0, 0, self.pad_len - num_of_embeddings), value=-1)

        ret = {
            "base_embeddings": base_embeddings,
            "attack_embeddings": attack_embeddings,
            "num_of_embeddings": num_of_embeddings,
            "success": success,
            "base_prompt": base_prompt,
            "attack_prompt": attack_prompt,
            "rephrasings": rephrasings,
            "seed": seed,
            "partition": partition,
            "task": task
        }

        return ret
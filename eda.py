import torch
from torch.nn import CosineSimilarity

#base_file = torch.load("/home/tyler/Gradskool/laser/LASER/dataset/attack_prompts/placement/attack_placement_generalization_Extendrearrange_42_dataset.pt")
#attack_file = torch.load("/home/tyler/Gradskool/laser/LASER/dataset/attack_prompts/combinatorial/attack_combinatorial_generalization_Extend_rearrange_42_dataset.pt")

attack_file = torch.load("LASER_Dataset/attack_prompts/placement/attack_placement_generalization_Extendrearrange_42_dataset.pt")
base_file = torch.load("LASER_Dataset/base_prompts/placement/base_placement_generalization_rearrange_42_dataset.pt")

for i in range(9):
    for j in range(150):
        attack, base = attack_file[i][j]["embedding"], base_file[0][j]["embedding"]
        k, _, _ = attack.shape
        att_embs = base.squeeze(dim=1)
        base_embs = attack.squeeze(dim=1)

        cos_f = CosineSimilarity(dim=0, eps=1e-6)
        cos_b = CosineSimilarity(dim=1, eps=1e-6)

        sim_flatten = cos_f(att_embs.flatten(), base_embs.flatten())
        sim_batch = cos_b(att_embs, base_embs)

        print("Base prompt: ", base_file[0][j]["base_prompt"])
        print("Attack prompt: ", attack_file[i][j]["attack_prompt"])
        print("Task: ", attack_file[i][j]["task"])
        print("Rephrasings: ", attack_file[i][j]["rephrasings"])
        print("Partition: ", attack_file[i][j]["partition"])
        print("Vision Attack: ", attack_file[i][j]["vis_atk"])
        print("Random Seed: ", attack_file[i][j]["seed"])
        print("Random Seed: ", attack_file[i][j]["seed"])
        print("success: ", attack_file[i][j]["success"], i, j)

        print("Cosine sim for flattening the embeddings:")
        print("embedding shapes: ", base_embs.flatten().shape, att_embs.flatten().shape)
        print("Flatten layer similarity: ", sim_flatten)
        
        print("Cosine sim per token embedding :")
        print("embedding shapes: ", base_embs.shape, att_embs.shape)
        print("Similarity per batch:")
        print(sim_batch)
        
        print()
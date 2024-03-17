# LASER

This project is forked from the original [[VIMA Repo]](https://github.com/vimalabs/VIMA)

As discussed in the paper [[On the Safety Concerns of Deploying LLMs/VLMs in Robotics]](https://arxiv.org/pdf/2402.10340.pdf),
new ways to prevent attacks on multimodal models are needed as they may have drastic consequences when deployed in a real world environment.

In this repo, we analyze the output embedding space of the T5 LLM in the VIMA pipeline to determine a way to align stealth attack prompt embeddings
to sucessful ones in order to prevent these types of attacks.




##### Team Members 

    Wen-Kai Chen

    Ashton Frias

    Nicholas Gray

    Abhinav Kotta

    Tyler VanderMate


### Milestones

- 1 Create instructions to get VIMA working
- 2a Replicate VIMA
- 3a generate attack prompts
- 3b see if base embeddings and attack embeddings are dissimilar using cosine similarity
- 4 replicate with attack prompts
- 5 add a MLP to the end of the T5 model encoding
- 6 train MLP with loss against attack prompts using base encoding as ground truth target
- 7 implement random noise into the target embedding space of the base prompt to adjust it slightly in order to learn mappings from various attack encodings to the area clustered around the base encoding
- 8 Plug the model back into the VIMA pipeline and compare results
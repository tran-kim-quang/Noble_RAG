import pickle
import torch
import os

avatar_path = 'c:/GithubRepo/Noble_RAG/livetalking/data/avatars/half-avatar'

with open(os.path.join(avatar_path, 'coords.pkl'), 'rb') as f:
    coords = pickle.load(f)
    print(f"Coords length: {len(coords)}")

with open(os.path.join(avatar_path, 'mask_coords.pkl'), 'rb') as f:
    mask_coords = pickle.load(f)
    print(f"Mask Coords length: {len(mask_coords)}")

latents = torch.load(os.path.join(avatar_path, 'latents.pt'), map_location='cpu')
print(f"Latents length: {len(latents)}")

# title_huggingface_emb.py
# Script can be used on Opuses for picking the best
# text embedder and the best dimension reduction method.

OPUS = False

import pandas as pd
from transformers import CLIPProcessor, CLIPModel
import torch
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import os
import json
from PIL import Image
from sklearn.decomposition import PCA

custom_cache = "/pio/scratch/1/i337942/huggingface_cache"

if OPUS:
    os.environ["HF_HOME"] = custom_cache
    os.environ["TRANSFORMERS_CACHE"] = custom_cache
    os.environ["HF_DATASETS_CACHE"] = custom_cache
    os.environ["HF_HUB_CACHE"] = custom_cache
os.environ["HF_HUB_TOKEN"] = open('./HUGGING_FACE_KEY.keys', 'r').read()
FILE_SYSTEM_PREFIX = './images/'

device = "cuda" if torch.cuda.is_available() else "cpu"

DATASET = 'Amazon_Sports_and_Outdoors'

META_PATH = f'../../{DATASET}/meta_{DATASET}.jsonl'
SAVE_PATH_PCA = f'../../{DATASET}/{DATASET}(MobileCLIP-S0-pca).image'
SAVE_PATH_KPCA = f'../../{DATASET}/{DATASET}(MobileCLIP-S0-kpca).image'
MAPPING_PATH = f'../../{DATASET}/item_reverse_mapping_{DATASET}.json'

model_name = "apple/MobileCLIP-S0"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# Change depending on your GPU :(
batch_size = 32

with open(META_PATH) as f:
    parent_asins = [json.loads(line)['parent_asin'] for line in f]

with open(MAPPING_PATH) as f:
    mapping_dict = json.load(f)

image_parent_asins = set()
embeddings = []
for i in tqdm(range(0, len(parent_asins), batch_size)):
    ids = parent_asins[i:i + batch_size]

    batch = []
    for item_id in ids:
        try:
            with Image.open(f"{FILE_SYSTEM_PREFIX}{item_id}.jpg") as im:
                batch.append(im.convert("RGB"))
            image_parent_asins.add(item_id)
        except FileNotFoundError:
            continue

    inputs = processor(images=batch, return_tensors="pt").to(device)

    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.cpu())

    del inputs
    torch.cuda.empty_cache()

no_image_parent_asins = set(parent_asins) - image_parent_asins

embeddings = torch.cat(embeddings, dim=0)
embeddings = embeddings.numpy()

# Do approx. KPCA

nys = Nystroem(kernel="cosine", n_components=128, random_state=42)

embeddings_128 = nys.fit_transform(embeddings)

df = pd.DataFrame(
    {'item_id:token': [mapping_dict[entry] for entry in image_parent_asins]}
)

df.rename(columns={'item_id:token' : 'ent_id:token'}, inplace=True)
df['ent_emb:float_seq'] = [' '.join(map(str, vec)) for vec in embeddings_128]

zero_rows = pd.DataFrame({
    'ent_id:token': [mapping_dict[entry] for entry in no_image_parent_asins],
    'ent_emb:float_seq': [' '.join(['0'] * 128) for _ in no_image_parent_asins]
})

df = pd.concat([df, zero_rows], ignore_index=True)

df.to_csv(SAVE_PATH_KPCA, sep="\t", index=False)

# Approximate correlation between reduced and original dataset
orig_sim = cosine_similarity(embeddings[:1000])
reduced_sim = cosine_similarity(embeddings_128[:1000])
correlation = np.corrcoef(orig_sim.flatten(), reduced_sim.flatten())[0, 1]

# ~0.9 is quite good
print("Similarity correlation:", correlation)

# Do PCA

pca = PCA(n_components=128)

embeddings_128 = pca.fit_transform(embeddings)

df = pd.DataFrame(
    {'item_id:token': [mapping_dict[entry] for entry in image_parent_asins]}
)

df.rename(columns={'item_id:token' : 'ent_id:token'}, inplace=True)
df['ent_emb:float_seq'] = [' '.join(map(str, vec)) for vec in embeddings_128]

zero_rows = pd.DataFrame({
    'ent_id:token': [mapping_dict[entry] for entry in no_image_parent_asins],
    'ent_emb:float_seq': [' '.join(['0'] * 128) for _ in no_image_parent_asins]
})

df = pd.concat([df, zero_rows], ignore_index=True)

df.to_csv(SAVE_PATH_PCA, sep="\t", index=False)

# Approximate correlation between reduced and original dataset
orig_sim = cosine_similarity(embeddings[:1000])
reduced_sim = cosine_similarity(embeddings_128[:1000])
correlation = np.corrcoef(orig_sim.flatten(), reduced_sim.flatten())[0, 1]

# ~0.9 is quite good
print("Similarity correlation:", correlation)
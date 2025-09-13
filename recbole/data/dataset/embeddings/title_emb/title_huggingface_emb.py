# title_huggingface_emb.py
# Script can be used on Opuses for picking the best
# text embedder and the best dimension reduction method.

OPUS = False

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import os

custom_cache = "/pio/scratch/1/i337942/huggingface_cache"

if OPUS:
    os.environ["HF_HOME"] = custom_cache
    os.environ["TRANSFORMERS_CACHE"] = custom_cache
    os.environ["HF_DATASETS_CACHE"] = custom_cache
    os.environ["HF_HUB_CACHE"] = custom_cache
os.environ["HF_HUB_TOKEN"] = open('./HUB_TOKEN.keys', 'r').read()

DATASET = 'Amazon_Sports_and_Outdoors'

ITEM_PATH = f'../../{DATASET}/{DATASET}.item'
SAVE_PATH = f'../../{DATASET}/{DATASET}3.title'

df = pd.read_csv(ITEM_PATH, delimiter='\t')
df = df[['item_id:token', 'title:token']]

titles = df['title:token'].astype('str').tolist()

model = SentenceTransformer(
    'BAAI/bge-large-en-v1.5',
    cache_folder=custom_cache if OPUS else None
)
batch_size = 512

embeddings = []
for i in tqdm(range(0, len(titles), batch_size)):
    batch = titles[i:i + batch_size]
    emb = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
    embeddings.extend(emb)

nys = Nystroem(kernel="cosine", n_components=128, random_state=42)

embeddings_128 = nys.fit_transform(embeddings)

df = df[['item_id:token']]

df.rename(columns={'item_id:token' : 'ent_id:token'}, inplace=True)
df['ent_emb:float_seq'] = [' '.join(map(str, vec)) for vec in embeddings_128]

df.to_csv(SAVE_PATH, sep="\t", index=False)

# Approximate correlation between reduced and original dataset
orig_sim = cosine_similarity(embeddings[:1000])
reduced_sim = cosine_similarity(embeddings_128[:1000])
correlation = np.corrcoef(orig_sim.flatten(), reduced_sim.flatten())[0, 1]

# ~0.9 is quite good
print("Similarity correlation:", correlation)

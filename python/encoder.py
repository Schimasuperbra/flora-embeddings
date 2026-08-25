from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import numpy as np

# Run this script from the repository root:
#   python encoder.py

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer(
    "intfloat/multilingual-e5-large-instruct",
    device=device,
)

input_file = "data/floras_0609_clean.csv"
output_file = "data/flora_emb_multilingual_0617_with_metadata.npz"

df = pd.read_csv(input_file)

# Encode rows in file order and WITHOUT deduplication: the downstream
# notebooks index the embedding matrix by row position, so the rows here
# must stay aligned with floras_0609_clean.csv.
flora_descriptions = (
    df["Description"]
    .fillna("")          # missing descriptions -> empty string (not the literal "nan")
    .astype(str)
    .tolist()
)

document_embeddings = model.encode(
    flora_descriptions,
    batch_size=32,
    convert_to_numpy=True,
    show_progress_bar=True,
    normalize_embeddings=True,
    device=device,
)

# Save the embedding matrix together with the species names, in the .npz
# format that the notebooks load:
#   np.load(output_file, allow_pickle=True)["embeddings"]
species_names = df["scientificName"].astype(str).values
np.savez(
    output_file,
    embeddings=document_embeddings,
    scientificName=species_names,
)

print(f"Saved embeddings {document_embeddings.shape} "
      f"for {len(species_names)} species to {output_file}")
